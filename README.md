# Action_recognition_tested
This is reposiotry to run Action Recognition model with multiple rois. 

Use Docker Image: 
note : This can run over docker images greater than DS version 6.1
sudo docker pull nvcr.io/nvidia/deepstream:6.4-triton-multiarch

Steps : 

- First Make the deepstream folder files for deepstream-test5-analytics
- Go to config
- run deepstream-test5-analytics -c deepstream_app_config_30.txt -t 
