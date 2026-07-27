## Install
```bash
conda create --name CardiacLab python=3.9 -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Install MONAI Label Extension in 3D Slicer:
* Click View -> Extensions Manager from the top menu.
* Search for monai and click INSTALL under MONAILabel.
* Restart 3D Slicer when prompted by the system.
* Go to Edit -> Application Settings -> Modules.
* Find MONAILabel in the modules list, drag and drop it into the Favorite Modules area at the bottom, and click OK to save.
* Click the MONAILabel icon on the toolbar to open the interface.

### Parameter ConfigurationSet 
* **Segmentation Target:** Open the \CardiacLabV2\run\segmentation.py file.
* Modify the parameter on line 7 based on your needs:
    * `segmentation_cardiac` for Whole Heart Muscle.
    * `segmentation_lvmyo` for Left Ventricular Myocardium.
    * `segmentation_heartcontour` for Heart Contour.
* Set Model Architecture: Open the \CardiacLabV2\run\config.toml file.
* Modify the model parameter on line 1. The available options are unetcnx_a1, testnet, or unetwic.
  
## Run Server
### By cmd
```bash
monailabel start_server \
--app radiology \
--studies "D:\tmp\data" \
--conf models segmentation_cardiac \
--conf network unetwic \
--conf --download_ckp_id <ID> \
--conf --target_spacing "0.7, 0.7, 1.0" \
--conf --spatial_size "128, 128, 128"\
--conf --intensity "-42, 423"
```

## By script
set config to run/config.toml.
```bash
model = 'unetwic'

[segmentation_cardiac.unetwic]
app = "radiology"
studies = "D:\\tmp\\data"
models = "segmentation_cardiac"
network = "unetwic"
download_ckp_id = "<ID>"
target_spacing = "0.7, 0.7, 1.0"
spatial_size = "128, 128, 128"
intensity = "-42, 423"
```
run run/segmentation.py.
```bash
python run/segmentation.py
```
* **Verify Connection:** Upon successful startup, you will see running on http://0.0.0.0:8000 in the terminal.Keep this window open.
* Run in 3D Slicer:
    * Click the MONAI Label icon, then click the refresh icon to connect to the server.
    * In the Strategy field, select first to load patients in order.
    * Click Next Sample to load the patient data.
    * Click Run to execute the segmentation.
    * After the segmentation is complete, click Submit Label to save the results.
    * Click Next Sample again to load the next patient data and repeat the process.
* Finish: Once all patients are labeled, you can close 3D Slicer and the terminal.
*  The predicted segmentation results are saved in the D:\tmp\data\labels\final folder.
