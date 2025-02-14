# Setup
```
pip install -U "huggingface_hub[cli]"
```
# Download Llama-Guard-3-1B
```
huggingface-cli download meta-llama/Llama-Guard-3-1B --local-dir Llama-Guard-3-1B  --token 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
huggingface-cli download meta-llama/Llama-Guard-3-8B --local-dir Llama-Guard-3-8B  --token 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
```
# Download dataset
```
huggingface-cli download dstc12/bot_adversarial_dialogue --repo-type dataset  --local-dir dstc12/bot_adversarial_dialogue --token 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
huggingface-cli download dstc12/ProsocialDialog --repo-type dataset  --local-dir dstc12/ProsocialDialog --token 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 
huggingface-cli download dstc12/dialogue_safety --repo-type dataset  --local-dir dstc12/dialogue_safety --token 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
```
