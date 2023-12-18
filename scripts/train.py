# Train Flash model
import flash
import torch
from flash.text import TextClassificationData, TextClassifier

"""
    Process:
    1. Create the DataModule
    2. Build the task
    3. Create the trainer and finetune the model
    4. Generate predictions for a few comments
    5. Save the model
"""

labels = {'law':[
            '刑法第11條', '毒品危害防制條例第10條', '刑法第47條', '刑事訴訟法第299條', '刑法第41條', '刑事訴訟法第454條',
            '毒品危害防制條例第18條', '刑事訴訟法第273條', '刑事訴訟法第449條', '刑法第38條', '刑法第51條', '刑事訴訟法第450條',
            '刑事訴訟法第455條', '刑法第10條', '毒品危害防制條例第4條', '刑法第62條', '毒品危害防制條例第11條', '刑法第56條',
            '毒品危害防制條例第19條', '刑法第55條', '毒品危害防制條例第17條', '刑事訴訟法第310條', '刑事訴訟法第368條',
            '刑事訴訟法第369條', '刑事訴訟法第364條'
            
        ], 'sentence':[
            '(425.0, 1335.0]', '(4.999, 90.0]', '(150.0, 180.0]','(90.0, 120.0]', '(1335.0, 10950.0]',
            '(300.0, 425.0]', '(180.0, 210.0]', '(240.0, 300.0]','(120.0, 150.0]', '(210.0, 240.0]'
        ]}

model_names = [
                'ckiplab/bert-base-chinese', # F1 66
                'bert-base-chinese', # About the same as ckiplab
                'schen/longformer-chinese-base-4096', 
                'IDEA-CCNL/Erlangshen-Longformer-330M',
                'Lowin_chinese-bigbird-wwm-base-4096',
                'prajjwal1/bert-tiny',
                ]

model_num = -1
max_length = 512
accumulate_grad_batches=2
max_epochs = 10
precision = "16"

for ind, item in enumerate(labels.items()):
    # Create the DataModule
    datamodule = TextClassificationData.from_csv(
        "main_body",
        item[1],
        train_file="./data/processed_all_drug_top_25_act_flash.csv",
        val_split=0.1,
        batch_size=64,
    )

    if item[0] == 'law':
        # continue
        model = TextClassifier(
        backbone=model_names[model_num],
        labels=datamodule.labels,
        multi_label=datamodule.multi_label,
        max_length=max_length
        )
    elif item[0] == 'sentence':
        # continue
        model = TextClassifier(
            backbone=model_names[model_num],
            labels=datamodule.labels,
            max_length=max_length,
            num_classes=len(labels['sentence'])
        )
    print(item[0])

    # Create the trainer and finetune the model
    trainer = flash.Trainer(max_epochs=max_epochs,\
                gpus=torch.cuda.device_count(),\
                accumulate_grad_batches=accumulate_grad_batches,\
                precision=precision)

    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # Save the model
    trainer.save_checkpoint(f"./models/{item[0]}_classification_{model_names[model_num].replace('/','-')}.pt")