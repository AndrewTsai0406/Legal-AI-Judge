# --coding:utf-8--
import os
import shutil

import gradio as gr
from fastapi import Request, FastAPI, Response

import torch
import flash
from flash.text import TextClassificationData, TextClassifier
from huggingface_hub import hf_hub_download

labels = {'law':[
            '刑法第11條', '毒品危害防制條例第10條', '刑法第47條', '刑事訴訟法第299條', '刑法第41條', '刑事訴訟法第454條',
            '毒品危害防制條例第18條', '刑事訴訟法第273條', '刑事訴訟法第449條', '刑法第38條', '刑法第51條', '刑事訴訟法第450條',
            '刑事訴訟法第455條', '刑法第10條', '毒品危害防制條例第4條', '刑法第62條', '毒品危害防制條例第11條', '刑法第56條',
            '毒品危害防制條例第19條', '刑法第55條', '毒品危害防制條例第17條', '刑事訴訟法第310條', '刑事訴訟法第368條',
            '刑事訴訟法第369條', '刑事訴訟法第364條'],
        'sentence':[
            '(425.0, 1335.0]', '(4.999, 90.0]', '(150.0, 180.0]','(90.0, 120.0]', '(1335.0, 10950.0]',
            '(300.0, 425.0]', '(180.0, 210.0]', '(240.0, 300.0]','(120.0, 150.0]', '(210.0, 240.0]']
            }

def load_classification_model():
    REPO_ID = "AndrewTsai0406/bert-tiny-law-article-classifier"
    MODEL_NAME = 'law_classification_prajjwal1-bert-tiny.pt'

    if os.path.isfile('./models/'+MODEL_NAME):
        print("File exists in the directory.")
    else:
        print("File does not exist in the directory.")
        hf_hub_download(repo_id=REPO_ID, filename=MODEL_NAME, local_dir_use_symlinks=False, local_dir='./models')
    model = TextClassifier.load_from_checkpoint('./models/'+MODEL_NAME)
    return model


def load_regression_model():
    REPO_ID = "AndrewTsai0406/bert-tiny-jail-sentence-classifier"
    MODEL_NAME = 'sentence_classification_prajjwal1-bert-tiny.pt'
    if os.path.isfile('./models/'+MODEL_NAME):
        print("File exists in the directory.")
    else:
        print("File does not exist in the directory.")
        hf_hub_download(repo_id=REPO_ID, filename=MODEL_NAME, local_dir_use_symlinks=False, local_dir='./models')
    model = TextClassifier.load_from_checkpoint('./models/'+MODEL_NAME)
    return model

model_classificaiton = load_classification_model()
model_regression = load_regression_model()
trainer = flash.Trainer(gpus=torch.cuda.device_count(), enable_checkpointing=False)
# trainer = flash.Trainer(gpus=torch.cuda.device_count(), accelerator="gpu", enable_checkpointing=False)

def gradio_model_inference(text):
    datamodule = TextClassificationData.from_lists(predict_data=[text],batch_size=4)
    predictions_law_articels = trainer.predict(model_classificaiton, datamodule = datamodule, output="labels")
    predictions_sentence = trainer.predict(model_regression, datamodule = datamodule, output="labels")
    pred_from, pred_to = labels['sentence'][int(predictions_sentence[0][0])].split(', ')
    predictions_sentence = 'Jail time: from '+f'{int(float(pred_from[1:]))}'+' days'+' to '+f'{int(float(pred_to[:-1]))}'+' days'
    return '\n'.join(predictions_law_articels[0][0]), predictions_sentence

demo = gr.Interface(fn=gradio_model_inference,
                    inputs=gr.Textbox(lines=10, label='Judgment', placeholder="Please give an document here for inference."),
                    outputs=[gr.Textbox(label='Law article prediction'),
                             gr.Textbox(label='Sentence prediction')],
                    title="AI Judge (ML Zoomcamp Project)",
                    description="""## A Legal Decision Support System
                            ### 1. AI-powered Decision Making
                            Utilizing state-of-the-art machine learning algorithms, the AI Judgment Predictor processes vast amounts of legal data to generate accurate predictions regarding potential legal outcomes. This includes identifying the specific laws relevant to a case and evaluating their application in light of precedent-setting decisions.
                            ### 2. Sentencing Projections
                            In addition to identifying applicable laws, the AI Judgment Predictor goes a step further by providing estimations of potential sentences for incarceration. By considering factors such as the severity of the offense, any aggravating or mitigating circumstances, and jurisdiction-specific sentencing guidelines, the system offers a valuable insight into the potential legal ramifications.""",
                    # theme=gr.themes.Monochrome(),
                    article="![Alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpZJRGjMclwPOjGNLaiGsKaloSP2CTz1Z4tg&usqp=CAU)",
                    examples=['理由本件原判決認定上訴人甲○○基於意圖營利之概括犯意，自民國八十六年四月上旬某日起至同年六月二十一日下午約四、五時止，先後在其位於雲林縣斗六市鎮○里鎮○路二七八巷十八號住處、雲林縣斗六市○○路某廟旁或同市○○路口戴凱勇所經營之「儂徠檳榔攤」、同市○○路五克拉ＭＴＶ店內等地，將已裝有少許海洛因粉末之注射針筒，以每支新台幣（下同）一千元之價格販賣予戴凱勇六次（在鎮北路及儂徠檳榔攤各二次，其餘各一次），每次販賣一支予戴凱勇施用（販毒所得財物六千元），嗣於八十六年六月二十一日晚上七時五十分許，戴凱勇為警查獲後供出其毒品來源係向上訴人所購，經警授意再度以呼叫器連絡上訴人購買毒品，嗣於同日晚上十一時許，上訴人與不知情之劉忠興二人前往上開「儂徠檳榔攤」欲販賣予戴凱勇時，經警方逮捕上訴人、劉忠興（已經第一審另案判處施用毒品罪刑確定）二人，並在上訴人身上扣得其供預備販賣所用已裝有海洛因粉末之注射針筒三支及毒品海洛因一小包（警方為檢驗方便，將三支針筒內之海洛因倒出分別裝入三小包夾鏈袋內，總共四小包海洛因送驗之驗餘淨重○‧一○公克）等情，因而撤銷第一審關於上訴人販賣毒品部分之科刑判決，改判仍論處上訴人連續販賣毒品（累犯）罪刑，固非無見'] )
demo.launch(share=True)

try:
    shutil.rmtree('./lightning_logs')
except FileNotFoundError:
    pass