language=("ar" "de" "en" "es" "fr" "ja" "pt" "zh")
for lang in "${language[@]}";do
    python LlamaGuard.py --input_file dstc12/bot_adversarial_dialogue/$lang
    echo "Done $lang"
done
