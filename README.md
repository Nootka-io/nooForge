# NooForge

Efficient tokenization and training of huggingface transformers models.

> **NOTE:** This is an alpa release (and I would barely call it that). The code works, however it's lacking any documentation, packaging, and requirements. Some more details and a usage example are available here: <https://github.com/getorca/stock_price_chat>

## Tokenization

Tokenization is done before training in a isolated step. This is done to effieciently pack the training samples, and create more seperation of concerns. It's largely handled by duckdb and uses Jinja2 templates for constructing samples.

Usage example: <https://github.com/getorca/stock_price_chat/blob/main/training_scripts/stc_config.yml>

### Templating

Uses PyYaml (YAML 1.1 support) to parse the YAML file and Jinja2 to render the templates.

See <https://yaml.org/spec/1.1/> for more info on yaml syntax.

See <https://jinja.palletsprojects.com/en/3.1.x/api/> for more on Jinja2 syntax.

## Training

Usage example: <https://github.com/getorca/stock_price_chat/blob/main/training_scripts/finetune_spc_01.sh>

## Citations / Credits

- flash attention llama patch <https://raw.githubusercontent.com/philschmid/deep-learning-pytorch-huggingface/b66a122e887ac8727d220a84e96dd48fee307490/training/utils/llama_patch.py>

## ToDos

- training
  - metrics
  - make eval optional
  - make flash attention optional
  - support 4 and 8 bit training
  - support other attention mechnisms, like xformers <https://github.com/facebookresearch/xformers>
  - dockerize training environment
- tokenization
  - more felxibility  
