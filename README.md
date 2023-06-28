CudaPBRT
======================

A Cuda Path Tracer by Mengxuan Huang

# Features

## BVH Acceleration

## Physically Based Materials

### Standford_Bunny Scene
| |Roughness = 0 | Roughness = 0.3 | Roughness = 0.6 | Roughness = 1 |
| ------------- |------------- |------------- | ------------- | ------------- |
| Metallic = 0|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/abae57bf-3fff-4cb2-9728-a0b083815650)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/e84e729f-a58e-4a09-803d-13b3eb932c83)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/f9d5d053-d53c-4e03-916b-618fd898e1d1)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/7c8e6074-9ec1-47d6-baa3-6c18bdcbaf69)|
| Metallic = 0.3|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/d90c26fe-9b09-4b14-b079-52f5dd52d633)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/225e7356-600a-426e-8763-5c5216257cd4)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/85b5d6cb-bef4-48e9-8476-c3e66e9f6487)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/dad2e30d-fff8-4b67-a310-32272c2f8e3f)|
| Metallic = 0.6|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/66e233f7-3ea0-4bcd-8d64-4fe555e00b71)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/0143f4e9-aef9-4367-82ac-ed35d20a5860)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/4bcb2f4d-e492-42d1-9cba-33e538ed42ee)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/ddb726e7-20c1-4ce0-816c-336874227524)|
| Metallic = 1|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/79bf23e1-0650-4bc4-8278-7353b91aa5b6)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/63761814-b4a6-4234-bd4b-4fdc31fbde5a)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/72791439-6589-4719-ae0e-94dc9d7ae512)|![Stanford_Bunny json_M20_it300](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/728b536f-be8b-4f0b-bd9e-a337dda3b1c3)|

## Len Camera

## Importance Sampling

## BRDF Importance

## Light Source Importance Sampling

## Resampled Importance Sampling

## Multiple Importance Sampling

## ReSTIR DI

# Results:
### CornellBox Scene (50 iterations)
| GBuffer preview (Normal) | Naive PT | Direct Lighting | MIS |
| ------------- |------------- |------------- | ------------- |
|![CornellBox json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/719e72c8-527e-4695-9b9d-1256daf38733)|![CornellBox json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/36a23fd0-e6fa-46bd-a029-2b9ed361d2d1)|![CornellBox json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/aef201c7-09c7-45bc-961f-afa59dec2e5a)|![CornellBox json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/3e253bc9-c82f-4d45-bc65-9bce0e1bd289)|

### Complex Shape Lights Scene(50 iterations)
| GBuffer preview (Normal) | Naive PT | Direct Lighting | MIS | 
| ------------- |------------- |------------- | ------------- |
|![CornellBox_MultiLights json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/609f8e57-9ac4-4672-a5a7-4f7cf4e2a9e0)|![CornellBox_MultiLights json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/20d49b5f-a09a-4224-8f4d-ba5c883482dd)|![CornellBox_MultiLights json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/cca75555-ac25-465c-9aa7-4a543fab8ecb)|![CornellBox_MultiLights json_M1_it50](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/a5ae439d-94ee-40db-841f-0008fbbd6cf2)|

### Complex Shape Lights Scene(5 iterations) Direct Lighting
| RIS (N = 1, M = 1) | RIS (N = 1, M = 20) | Temporal Reuse (N = 1, M = 20) | Spatial Reuse (N = 1, M = 20) | ReSTIR DI (N = 1, M = 20)|
| ------------- |------------- |------------- | ------------- | ------------- |
|![CornellBox_MultiLights json_M1_it5](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/b4403299-71f1-4abb-b742-89e8f1d5701b)|![CornellBox_MultiLights json_M20_it5](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/c6f552fb-cb9b-436e-ad76-bc1b0542584d)|![CornellBox_MultiLights json_M20_it5](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/7b77c9f9-d1eb-4391-bf89-94a3775b3c2f)|![CornellBox_MultiLights json_M20_it5](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/576fe7f7-678d-4fd1-9a59-246b13d8788a)|![CornellBox_MultiLights json_M20_it5](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/c8274785-c0ed-498b-9778-f6bdc6bc9662)|

### Demo Scenes
| Camera (MIS 700 iterations) | Castle (ReSTIR DI 100 iterations) |
| ------------- |------------- |
|![Camera json_M1_it773](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/0159d4f4-5933-479b-b0ab-50d797ee8a04)|![Castle json_M1_it100](https://github.com/MengxuanHUANG/CudaPBRT/assets/53021701/7b31a931-417e-4d46-b729-0582f071d757)|
