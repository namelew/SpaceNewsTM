# SpaceNewsTM
Creating topics from the news in Space News Dataset using MGLDA
## Build
1. Download the Space News Kaggle Dataset and store in ./data
2. Create the virtual envoriment
    * Linux
    ```
    python3 -m venv myenv
    source myenv/bin/activate
    pip3 install -r ./requirements.txt
    python3 ./utils/Internal.py
    ```
    * Windows
    ```
    python -m venv myenv
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
    .\myenv\Scripts\activate
    pip install -r ./requirements.txt
    python ./utils/Internal.py
    ```
## Run
PS:
Run
```
Set-ExecutionPolicy Restricted
```
to undo
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```