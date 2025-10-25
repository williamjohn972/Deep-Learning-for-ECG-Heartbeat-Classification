import json
import os 

import pandas as pd

def log_to_json(filepath,content):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath,mode="w") as f:
        json.dump(content,f,indent=4)

def log_to_csv(filepath, content):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    content_df = pd.DataFrame(content)
    content_df.to_csv(filepath, index=False)
