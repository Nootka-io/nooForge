from typing import Any
from jinja2 import Environment, FileSystemLoader
import yaml
import argparse
import duckdb as ddb
from duckdb.typing import *
from transformers import AutoTokenizer
from datasets import Dataset
import logging


logging.basicConfig(level=logging.INFO)


class NooTokenizer:
    '''
    Tokenizes datasets for training. 
    It is recommended to split off eval data before using this tokenizer as it creates a single tokenized input for training.
    Saves the dataset with huggingface datasets.
    Also optionally updates the tokenizer with new special tokens. 
    '''
    def __init__(self, args):
        self.args = args
        self.config = yaml.safe_load(open(args.config))
        self.max_length = self.config['max_length'] - round(self.config['max_length'] * 0.025)
        self.padded_max_length = self.config['max_length']
        self.pad_token_id = None
        
    def __call__(self):
        # print(self.config)

        ddb.sql('SET enable_progress_bar=true;')
        
        for ds_config in self.config['Datasets']:
            
            template_config = self.config['Templates'][ds_config.get('template')]
            
            needed_column_str = ', '.join([x for x in ds_config.get('template_mapping').values()])
            
            tokenizer = self.load_tokenizer()
            
            if self.config['tokenizer']['name']:
                tokenizer.save_pretrained(self.config['tokenizer']['name'])            
            
            self.pad_token_id = tokenizer.pad_token_id
            
            # open the cleaned dataset
            # ToDo: add support for other file formats
            logging.info('Loading dataset: %s', ds_config['file'])
            i_db = ddb.read_json(
                ds_config['file'], 
                format='newline_delimited'
            )
            
            def tokenize_udf(input_list: list[str]) -> list[int]:
                '''
                formats and tokenizes the input list to an input
                '''
                sample = dict(zip(needed_column_str.split(', '), input_list))
                environment = Environment(loader=FileSystemLoader("templates/"))
                template = environment.get_template(template_config.get('file'))
                template_mapping = ds_config.get('template_mapping')
                for k, v in template_mapping.items():
                    sample[k] = sample.get(v)
                if 'template_args' in template_config:
                    sample = {**sample, **template_config.get('template_args')}
                formated_prompt = template.render(sample)
                tokenized_prompt = tokenizer(formated_prompt)
            
                return tokenized_prompt['input_ids']
            
            ddb.create_function('tokenize', tokenize_udf)

            # format and tokenize the dataset
            logging.info(f'Tokenizing dataset')
            tokenized_db = ddb.sql(f"""
                SELECT 
                    tokenize([{needed_column_str}]) as input_ids,
                    len(input_ids) as token_count
                FROM i_db
            """)
            
            logging.info(f'Starting grouping')
            self.group_dataset()  # group the dataset by token count
            
            # ToDo: handle training data that's longer than `max_length` by chunking it with an overlap and only adding eos string to the last chunk 
            
            # recursively split the groups that are longer than `max_length`
            while True:
                # this can take a long time with small samples
                logging.info(f'runnning recurisve grouping')
                
                # breakpoint()
                                
                updated_groups = self.split_overs()
                
                ddb.sql(f'DROP TABLE IF EXISTS grouped_data2;')
                
                grouped_db = ddb.sql(f"""
                    CREATE TABLE grouped_data2 AS SELECT * FROM (
                        SELECT 
                            *,
                            SUM(token_count) OVER(PARTITION BY group_flag) as group_token_count,
                            CAST(CASE WHEN group_token_count >= {self.max_length} THEN group_token_count - {self.max_length} ELSE 0 END AS INTEGER) AS over_by, 
                        FROM (
                            SELECT
                                grouped_data._id,
                                grouped_data.input_ids,
                                grouped_data.token_count,         
                                COALESCE(updated_groups.group_flag, grouped_data.group_flag) as group_flag,  
                            FROM grouped_data LEFT JOIN updated_groups ON grouped_data._id = updated_groups._id
                        )
                    );
                """)
                
                ddb.sql('DROP TABLE IF EXISTS grouped_data;')
                ddb.sql('CREATE TABLE grouped_data AS SELECT * FROM (SELECT * FROM grouped_data2);')
                
                has_rows = ddb.sql(f'SELECT * FROM grouped_data WHERE group_token_count > {self.max_length} LIMIT 1').fetchone()
                
                if has_rows:
                    continue
                else: 
                    break
            
            # append rows grouped rows togeather with EOS and PAD tokens, and create the attention mask
            logging.info(f'Creating final group with padding and creating attention mask')
            self.make_final_group()
            
            # save dataset for training
            logging.info(f'Saving hugging face dataset for training')
            arrow_ds = ddb.sql('SELECT input_ids_padded as labels, input_ids_padded as input_ids, attention_mask FROM tokenized_data ORDER BY RANDOM()').arrow()
            ds = Dataset(arrow_ds)
            ds.save_to_disk(self.config['tokenized_ds'])
            
            training_rows = len(ddb.sql('SELECT * FROM grouped_data').fetchall())
            token_count = ddb.sql('SELECT SUM(token_count) FROM grouped_data').fetchone()[0]
            
            return token_count, training_rows

    def group_dataset(self):
        '''
        groups the dataset by token count into groups around `max_length`
        '''        
        grouped_db = ddb.sql(f"""
            CREATE TABLE grouped_data AS SELECT * FROM (
            WITH cte AS (
                SELECT
                    input_ids,
                    token_count,
                    SUM(token_count) OVER (ORDER BY input_ids) AS cumulative_sum
                FROM
                    tokenized_db
                ORDER BY token_count DESC
            )
            SELECT
                row_number() OVER () as _id,
                input_ids,
                token_count,
                SUM(token_count) OVER (PARTITION BY group_flag) AS group_token_count,
                CAST(CASE WHEN group_token_count >= {self.max_length} THEN group_token_count - {self.max_length} ELSE 0 END AS INTEGER) AS over_by,
                group_flag,
            FROM (
                SELECT
                    input_ids,
                    token_count,
                    cumulative_sum,       
                    CASE WHEN cumulative_sum <= {self.max_length} THEN 0 ELSE cumulative_sum // {self.max_length} END AS group_flag,
                FROM
                    cte
            ) AS grouped_data
            );
        """)        
        return grouped_db

    def split_overs(self, ldb = 'grouped_data'):
        '''
        splits the groups that are longer than `max_length`
        is called recursively until all groups are less than `max_length`
        '''
        
        max_flag = max_flag = ddb.sql('SELECT * FROM grouped_data').max('group_flag').fetchone()
        
        updated_groups = ddb.sql(f"""        
            WITH cte AS (
                SELECT * FROM (
                    SELECT 
                        FIRST_VALUE(_id) OVER(
                            PARTITION BY group_flag ORDER BY ABS(token_count - over_by) ASC
                        ) AS _id 
                    FROM {ldb}
                    WHERE over_by > 0
                ) GROUP BY _id
            ),
            cte2 AS (
                SELECT 
                    cte._id as _id, 
                    input_ids, 
                    token_count, 
                    group_token_count,
                    SUM(token_count) OVER (ORDER BY input_ids) AS cumulative_sum 
                FROM {ldb} inner join cte on {ldb}._id = cte._id 
            )
            SELECT
                _id,
                input_ids,
                token_count,
                SUM(token_count) OVER (PARTITION BY group_flag) AS group_token_count,
                CAST(CASE WHEN group_token_count >= {self.max_length} THEN group_token_count - {self.max_length} ELSE 0 END AS INTEGER) AS over_by,
                group_flag,
            FROM (
                SELECT
                    _id,
                    input_ids,
                    token_count,     
                    CASE WHEN cumulative_sum <= {self.max_length} THEN ({max_flag[0]} + 1) ELSE ({max_flag[0]} + 1 + cumulative_sum // {self.max_length}) END AS group_flag,
                FROM
                    cte2
            )
        """)     
        return updated_groups
   
    def make_final_group(self):
        '''
        Creates the final grouping with padding and attention mask
        '''
        ddb.sql(f"""
            CREATE TABLE tokenized_data AS SELECT * FROM (
                SELECT
                    input_ids,
                    LEN(input_ids) as token_count,
                    CAST({self.padded_max_length} - token_count AS INTEGER) AS pad_count,
                    list_resize(input_ids, {self.padded_max_length}, {self.pad_token_id}) AS input_ids_padded,
                    list_resize(list_resize(list_value(1), token_count, 1), {self.padded_max_length}, 0) AS attention_mask
                FROM (
                    SELECT 
                        flatten(LIST(input_ids)) as input_ids 
                    FROM grouped_data 
                    GROUP BY group_flag
                )
            )
        """)
        
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model'], 
            add_eos_token=True,
            model_max_length=self.padded_max_length,
        )
        
        if self.config['tokenizer']['additional_special_tokens']:            
            special_tokens_dict = {'additional_special_tokens': self.config['tokenizer']['additional_special_tokens']}
            tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.pad_token = self.config['tokenizer']['pad_token']
        
        return tokenizer
        
    def tokenize(self, text):
        pass
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    
    tokenize = NooTokenizer(args=args)
    token_count, training_rows = tokenize()
    
    print('DONE Tokenization!')
    print(f'Token count: {token_count}')
    print(f'Training rows: {training_rows}')
    print('If you\'ve added special tokens make sure to copy the tokenizer to the model before training')

