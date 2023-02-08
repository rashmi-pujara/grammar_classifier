from flask import Flask, request
import torch
import traceback
import joblib
from sentence_transformers import SentenceTransformer, models
from transformers import T5ForConditionalGeneration, T5Tokenizer


app = Flask(__name__)

auth_token = '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'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'rashmi-pujara/grammar_suggestion'
tokenizer = T5Tokenizer.from_pretrained(model_name, use_auth_token=auth_token)
model = T5ForConditionalGeneration.from_pretrained(model_name, use_auth_token=auth_token).to(torch_device)

model_name_que_check = 'random_forest.joblib'
rf_que_check = joblib.load(model_name_que_check)

def createEmbedder():
    embedding_method = "gsarti/scibert-nli"
    word_embedding_model = models.Transformer(embedding_method, max_seq_length=512, do_lower_case=True)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())    
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

embedder = createEmbedder()

@app.route("/api/correct_grammar_suggestion", methods=["POST"])
def correct_grammar_suggestion(num_return_sequences=2, model=model):
    try:
        input = request.form["input"]
        temp_input = input
        qa = question_or_not_check(temp_input)
        if(qa["output"] == 1 and input[-1]!='?'):
            temp_input += "?"
        batch = tokenizer([temp_input], truncation=True, padding='max_length',
                        max_length=64, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch, max_length=64, num_beams=4,
                                    num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        grammar_corrected = 1
        if temp_input == tgt_text[0] or temp_input == tgt_text[1]:
            grammar_corrected = 0
        return {"opration": "success", "grammar_corrected" : grammar_corrected,"input": input, "output": tgt_text}
        
    except Exception as e:
        return {"opreation": "failed", "error": traceback.format_exc()}


@app.route("/api/question_or_not_check", methods=["POST"], defaults={'sentence': None})
def question_or_not_check(sentence):
    try:
        if(sentence == None):
            sentence = request.form["sentence"]
        emb = embedder.encode(sentence)
        output = rf_que_check.predict([emb])[0]
        return {"opration": "success", "input": sentence, "output": str(output)}
    except Exception as e:
        return {"opreation": "failed", "error": traceback.format_exc()}
