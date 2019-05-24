# 3D Machine Learning
In recent years, tremendous amount of progress is being made in the field of 3D Machine Learning, which is an interdisciplinary field that fuses computer vision, computer graphics and machine learning. This repo is derived from my study notes and will be used as a place for triaging new research papers. 

I'll use the following icons to differentiate 3D representations:
* :camera: Multi-view Images
* :space_invader: Volumetric
* :game_die: Point Cloud
* :gem: Polygonal Mesh
* :pill: Primitive-based

## Get Involved
To contribute to this Repo, you may add content through pull requests or open an issue to let me know. 

:star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:<br>
We have also created a Slack workplace for people around the globe to ask questions, share knowledge and facilitate collaborations. Together, I'm sure we can advance this field as a collaborative effort. Join the community with [this link](https://join.slack.com/t/3d-machine-learning/shared_invite/enQtMzUyMTgyNzgwOTgzLWIzY2M3MTQ1ODgwOWEwMGU3MWYxMThhOWQzZGY4OTdhM2VlYTc2N2FmNGVmMzE0MGJlNjg1NjA5OTRhNzlkOWQ).
<br>:star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:  :star:

## Table of Contents
- [Courses](#courses)
- [Datasets](#datasets)
  - [3D Models](#3d_models)
  - [3D Scenes](#3d_scenes)
- [3D Pose Estimation](#pose_estimation)
- [Single Object Classification](#single_classification)
- [Multiple Objects Detection](#multiple_detection)
- [Scene/Object Semantic Segmentation](#segmentation)
- [3D Geometry Synthesis/Reconstruction](#3d_synthesis)
  - [Parametric Morphable Model-based methods](#3d_synthesis_model_based)
  - [Part-based Template Learning methods](#3d_synthesis_template_based)
  - [Deep Learning Methods](#3d_synthesis_dl_based)
- [Texture/Material Analysis and Synthesis](#material_synthesis)
- [Style Learning and Transfer](#style_transfer)
- [Scene Synthesis/Reconstruction](#scene_synthesis)
- [Scene Understanding](#scene_understanding)

<a name="courses" />

## Available Courses
[Stanford CS231A: Computer Vision-From 3D Reconstruction to Recognition (Winter 2018)](http://web.stanford.edu/class/cs231a/)

[UCSD CSE291-I00: Machine Learning for 3D Data (Winter 2018)](https://cse291-i.github.io/index.html)

[Stanford CS468: Machine Learning for 3D Data (Spring 2017)](http://graphics.stanford.edu/courses/cs468-17-spring/)

[MIT 6.838: Shape Analysis (Spring 2017)](http://groups.csail.mit.edu/gdpgroup/6838_spring_2017.html)

[Princeton COS 526: Advanced Computer Graphics  (Fall 2010)](https://www.cs.princeton.edu/courses/archive/fall10/cos526/syllabus.php)

[Princeton CS597: Geometric Modeling and Analysis (Fall 2003)](https://www.cs.princeton.edu/courses/archive/fall03/cs597D/)

[Geometric Deep Learning](http://geometricdeeplearning.com/)

[Paper Collection for 3D Understanding](https://www.cs.princeton.edu/courses/archive/spring15/cos598A/cos598A.html#Estimating)

[CreativeAI: Deep Learning for Graphics](http://geometry.cs.ucl.ac.uk/creativeai/)

<a name="datasets" />

## Datasets
To see a survey of RGBD datasets, check out Michael Firman's [collection](http://www0.cs.ucl.ac.uk/staff/M.Firman//RGBDdatasets/) as well as the associated paper, [RGBD Datasets: Past, Present and Future](https://arxiv.org/pdf/1604.00999.pdf). Point Cloud Library also has a good dataset [catalogue](http://pointclouds.org/media/). 

<a name="3d_models" />

### 3D Models
<b>Princeton Shape Benchmark (2003)</b> [[Link]](http://shape.cs.princeton.edu/benchmark/)
<br>1,814 models collected from the web in .OFF format. Used to evaluating shape-based retrieval and analysis algorithms.
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Princeton%20Shape%20Benchmark%20(2003).jpeg" /></p>

<b>Dataset for IKEA 3D models and aligned images (2013)</b> [[Link]](http://ikea.csail.mit.edu/)
<br>759 images and 219 models including Sketchup (skp) and Wavefront (obj) files, good for pose estimation.
<p align="center"><img width="50%" src="http://ikea.csail.mit.edu/web_img/ikea_object.png" /></p>

<b>Open Surfaces: A Richly Annotated Catalog of Surface Appearance (SIGGRAPH 2013)</b> [[Link]](http://opensurfaces.cs.cornell.edu/)
<br>OpenSurfaces is a large database of annotated surfaces created from real-world consumer photographs. Our annotation framework draws on crowdsourcing to segment surfaces from photos, and then annotate them with rich surface properties, including material, texture and contextual information.
<p align="center"><img width="50%" src="http://opensurfaces.cs.cornell.edu/static/img/teaser4-web.jpg" /></p>

<b>PASCAL3D+ (2014)</b> [[Link]](http://cvgl.stanford.edu/projects/pascal3d.html)
<br>12 categories, on average 3k+ objects per category, for 3D object detection and pose estimation.
<p align="center"><img width="50%" src="http://cvgl.stanford.edu/projects/pascal3d+/pascal3d.png" /></p>

<b>ModelNet (2015)</b> [[Link]](http://modelnet.cs.princeton.edu/#)
<br>127915 3D CAD models from 662 categories
<br>ModelNet10: 4899 models from 10 categories
<br>ModelNet40: 12311 models from 40 categories, all are uniformly orientated
<p align="center"><img width="50%" src="http://3dvision.princeton.edu/projects/2014/ModelNet/thumbnail.jpg" /></p>

<b>ShapeNet (2015)</b> [[Link]](https://www.shapenet.org/)
<br>3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated.
<br>ShapeNetCore [[Link]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.
<p align="center"><img width="50%" src="http://msavva.github.io/files/shapenet.png" /></p>

<b>A Large Dataset of Object Scans (2016)</b> [[Link]](http://redwood-data.org/3dscan/index.html)
<br>10K scans in RGBD + reconstructed 3D models in .PLY format.
<p align="center"><img width="50%" src="http://redwood-data.org/3dscan/img/teaser.jpg" /></p>

<b>ObjectNet3D: A Large Scale Database for 3D Object Recognition (2016)</b> [[Link]](http://cvgl.stanford.edu/projects/objectnet3d/)
<br>100 categories, 90,127 images, 201,888 objects in these images and 44,147 3D shapes. 
<br>Tasks: region proposal generation, 2D object detection, joint 2D detection and 3D object pose estimation, and image-based 3D shape retrieval
<p align="center"><img width="50%" src="http://cvgl.stanford.edu/projects/objectnet3d/ObjectNet3D.png" /></p>

<b>Thingi10K: A Dataset of 10,000 3D-Printing Models (2016)</b> [[Link]](https://ten-thousand-models.appspot.com/)
<br>10,000 models from featured “things” on thingiverse.com, suitable for testing 3D printing techniques such as structural analysis , shape optimization, or solid geometry operations.
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DRbxWnqXkAEEH0g.jpg:large" /></p>

<b>ABC: A Big CAD Model Dataset For Geometric Deep Learning</b> [[Link]](https://cs.nyu.edu/~zhongshi/publication/abc-dataset/)[[Paper]](https://arxiv.org/abs/1812.06216)
<br>This work introduce a dataset for geometric deep learning consisting of over 1 million individual (and high quality) geometric models, each associated with accurate ground truth information on the decomposition into patches, explicit sharp feature annotations, and analytic differential properties.<br>
<p align="center"><img width="50%" src="https://cs.nyu.edu/~zhongshi/img/abc-dataset.png" /></p>

<a name="3d_scenes" />

### 3D Scenes
<b>NYU Depth Dataset V2 (2012)</b> [[Link]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
<br>1449 densely labeled pairs of aligned RGB and depth images from Kinect video sequences for a variety of indoor scenes.
<p align="center"><img width="50%" src="https://cs.nyu.edu/~silberman/images/nyu_depth_v2_labeled.jpg" /></p>

<b>SUNRGB-D 3D Object Detection Challenge</b> [[Link]](http://rgbd.cs.princeton.edu/challenge.html)
<br>19 object categories for predicting a 3D bounding box in real world dimension
<br>Training set: 10,355 RGB-D scene images, Testing set: 2860 RGB-D images
<p align="center"><img width="50%" src="http://rgbd.cs.princeton.edu/3dbox.png" /></p>

<b>SceneNN (2016)</b> [[Link]](http://www.scenenn.net/)
<br>100+ indoor scene meshes with per-vertex and per-pixel annotation.
<p align="center"><img width="50%" src="https://cdn-ak.f.st-hatena.com/images/fotolife/r/robonchu/20170611/20170611155625.png" /></p>

<b>ScanNet (2017)</b> [[Link]](http://www.scan-net.org/)
<br>An RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.
<p align="center"><img width="50%" src="http://www.scan-net.org/img/annotations.png" /></p>

<b>Matterport3D: Learning from RGB-D Data in Indoor Environments (2017)</b> [[Link]](https://niessner.github.io/Matterport/)
<br>10,800 panoramic views (in both RGB and depth) from 194,400 RGB-D images of 90 building-scale scenes of private rooms. Instance-level semantic segmentations are provided for region (living room, kitchen) and object (sofa, TV) categories. 
<p align="center"><img width="50%" src="https://niessner.github.io/Matterport/teaser.png" /></p>

<b>SUNCG: A Large 3D Model Repository for Indoor Scenes (2017)</b> [[Link]](http://suncg.cs.princeton.edu/)
<br>The dataset contains over 45K different scenes with manually created realistic room and furniture layouts. All of the scenes are semantically annotated at the object level.
<p align="center"><img width="50%" src="http://suncg.cs.princeton.edu/figures/data_full.png" /></p>

<b>MINOS: Multimodal Indoor Simulator (2017)</b> [[Link]](https://github.com/minosworld/minos)
<br>MINOS is a simulator designed to support the development of multisensory models for goal-directed navigation in complex indoor environments. MINOS leverages large datasets of complex 3D environments and supports flexible configuration of multimodal sensor suites. MINOS supports SUNCG and Matterport3D scenes.
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2017/12/MINOS.jpg" /></p>

<b>Facebook House3D: A Rich and Realistic 3D Environment (2017)</b> [[Link]](https://github.com/facebookresearch/House3D)
<br>House3D is a virtual 3D environment which consists of 45K indoor scenes equipped with a diverse set of scene types, layouts and objects sourced from the SUNCG dataset. All 3D objects are fully annotated with category labels. Agents in the environment have access to observations of multiple modalities, including RGB images, depth, segmentation masks and top-down 2D map views.
<p align="center"><img width="50%" src="https://user-images.githubusercontent.com/1381301/33509559-87c4e470-d6b7-11e7-8266-27c940d5729a.jpg" /></p>

<b>HoME: a Household Multimodal Environment (2017)</b> [[Link]](https://home-platform.github.io/)
<br>HoME integrates over 45,000 diverse 3D house layouts based on the SUNCG dataset, a scale which may facilitate learning, generalization, and transfer. HoME is an open-source, OpenAI Gym-compatible platform extensible to tasks in reinforcement learning, language grounding, sound-based navigation, robotics, multi-agent learning.
<p align="center"><img width="50%" src="https://home-platform.github.io/assets/overview.png" /></p>

<b>AI2-THOR: Photorealistic Interactive Environments for AI Agents</b> [[Link]](http://ai2thor.allenai.org/)
<br>AI2-THOR is a photo-realistic interactable framework for AI agents. There are a total 120 scenes in version 1.0 of the THOR environment covering four different room categories: kitchens, living rooms, bedrooms, and bathrooms. Each room has a number of actionable objects.
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/AI2-Thor.jpeg" /></p>

<b>UnrealCV: Virtual Worlds for Computer Vision (2017)</b> [[Link]](http://unrealcv.org/)[[Paper]](http://www.idm.pku.edu.cn/staff/wangyizhou/papers/ACMMM2017_UnrealCV.pdf)
<br>An open source project to help computer vision researchers build virtual worlds using Unreal Engine 4.
<p align="center"><img width="50%" src="http://unrealcv.org/images/homepage_teaser.png" /></p>

<b>Gibson Environment: Real-World Perception for Embodied Agents (2018 CVPR) </b> [[Link]](http://gibsonenv.stanford.edu/)
<br>This platform provides RGB from 1000 point clouds, as well as multimodal sensor data: surface normal, depth, and for a fraction of the spaces, semantics object annotations. The environment is also RL ready with physics integrated. Using such datasets can further narrow down the discrepency between virtual environment and real world.
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Gibson%20Environment-%20Real-World%20Perception%20for%20Embodied%20Agents%20(2018%20CVPR)%20.jpeg" /></p>

<b>InteriorNet: Mega-scale Multi-sensor Photo-realistic Indoor Scenes Dataset</b> [[Link]](https://interiornet.org/)
<br>System Overview: an end-to-end pipeline to render an RGB-D-inertial benchmark for large scale interior scene understanding and mapping. Our dataset contains 20M images created by pipeline: (A) We collect around 1 million CAD models provided by world-leading furniture manufacturers. These models have been used in the real-world production. (B) Based on those models, around 1,100 professional designers create around 22 million interior layouts. Most of such layouts have been used in real-world decorations. (C) For each layout, we generate a number of configurations to represent different random lightings and simulation of scene change over time in daily life. (D) We provide an interactive simulator (ViSim) to help for creating ground truth IMU, events, as well as monocular or stereo camera trajectories including hand-drawn, random walking and neural network based realistic trajectory. (E) All supported image sequences and ground truth.
<p align="center"><img width="50%" src="https://interiornet.org/items/InteriorNet.jpg" /></p>

<b>Semantic3D</b>[[Link]](http://www.semantic3d.net/)
<br>Large-Scale Point Cloud Classification Benchmark, which provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total, and also covers a range of diverse urban scenes.
<p align="center"><img width="50%" src="http://www.semantic3d.net/img/full_resolution/sg27_8.jpg" /></p>

<a name="pose_estimation" />

## 3D Pose Estimation
<b>Category-Specific Object Reconstruction from a Single Image (2014)</b> [[Paper]](https://people.eecs.berkeley.edu/~akar/categoryshapes.pdf)
<p align="center"><img width="50%" src="http://people.eecs.berkeley.edu/~akar/basisshapes_highres.png" /></p>

<b>Viewpoints and Keypoints (2015)</b> [[Paper]](https://people.eecs.berkeley.edu/~shubhtuls/papers/cvpr15vpsKps.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Viewpoints%20and%20Keypoints.jpeg" /></p>

<b>Render for CNN: Viewpoint Estimation in Images Using CNNs Trained with Rendered 3D Model Views (2015 ICCV)</b> [[Paper]](https://shapenet.cs.stanford.edu/projects/RenderForCNN/)
<p align="center"><img width="50%" src="https://shapenet.cs.stanford.edu/projects/RenderForCNN/images/teaser.jpg" /></p>

<b>PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization (2015)</b> [[Paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf)
<p align="center"><img width="50%" src="http://mi.eng.cam.ac.uk/projects/relocalisation/images/map.png" /></p>

<b>Modeling Uncertainty in Deep Learning for Camera Relocalization (2016)</b> [[Paper]](https://arxiv.org/pdf/1509.05909.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Modeling%20Uncertainty%20in%20Deep%20Learning%20for%20Camera%20Relocalization.jpeg" /></p>

<b>Robust camera pose estimation by viewpoint classification using deep learning (2016)</b> [[Paper]](https://link.springer.com/article/10.1007/s41095-016-0067-z)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Robust%20camera%20pose%20estimation%20by%20viewpoint%20classification%20using%20deep%20learning.jpeg" /></p>

<b>Geometric loss functions for camera pose regression with deep learning (2017 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1704.00390.pdf)
<p align="center"><img width="50%" src="http://mi.eng.cam.ac.uk/~cipolla/images/pose-net.png" /></p>

<b>Generic 3D Representation via Pose Estimation and Matching (2017)</b> [[Paper]](http://3drepresentation.stanford.edu/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Generic%203D%20Representation%20via%20Pose%20Estimation%20and%20Matching.jpeg" /></p>

<b>3D Bounding Box Estimation Using Deep Learning and Geometry (2017)</b> [[Paper]](https://arxiv.org/pdf/1612.00496.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/3D%20Bounding%20Box%20Estimation%20Using%20Deep%20Learning%20and%20Geometry.png" /></p>

<b>6-DoF Object Pose from Semantic Keypoints (2017)</b> [[Paper]](https://www.seas.upenn.edu/~pavlakos/projects/object3d/)
<p align="center"><img width="50%" src="https://www.seas.upenn.edu/~pavlakos/projects/object3d/files/object3d-teaser.png" /></p>

<b>Relative Camera Pose Estimation Using Convolutional Neural Networks (2017)</b> [[Paper]](https://arxiv.org/pdf/1702.01381.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Relative%20Camera%20Pose%20Estimation%20Using%20Convolutional%20Neural%20Networks.png" /></p>

<b>3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions (2017)</b> [[Paper]](http://3dmatch.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://3dmatch.cs.princeton.edu/img/overview.jpg" /></p>

<b>Single Image 3D Interpreter Network (2016)</b> [[Paper]](http://3dinterpreter.csail.mit.edu/) [[Code]](https://github.com/jiajunwu/3dinn)
<p align="center"><img width="50%" src="http://3dinterpreter.csail.mit.edu/images/spotlight_3dinn_large.jpg" /></p>

<b>Multi-view Consistency as Supervisory Signal  for Learning Shape and Pose Prediction (2018 CVPR)</b> [[Paper]](https://shubhtuls.github.io/mvcSnP/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/mvcSnP/resources/images/teaser.png" /></p>

<b>PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes (2018)</b> [[Paper]](https://rse-lab.cs.washington.edu/projects/posecnn/)
<p align="center"><img width="50%" src="https://yuxng.github.io/PoseCNN.png" /></p>

<b>Feature Mapping for Learning Fast and Accurate 3D Pose Inference from Synthetic Images (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1712.03904.pdf)
<p align="center"><img width="40%" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnpyajEhbhrPMc0YpEQzqE8N9E7CW_EVWYA3Bxg46oUEYFf9XvkA" /></p>

<b>Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling (2018 CVPR)</b> [[Paper]](http://pix3d.csail.mit.edu/)
<p align="center"><img width="50%" src="http://pix3d.csail.mit.edu/images/spotlight_pix3d.jpg" /></p>

<b>3D Pose Estimation and 3D Model Retrieval for Objects in the Wild (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1803.11493.pdf)
<p align="center"><img width="50%" src="https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/team_lepetit/images/grabner/pose_retrieval_overview.png" /></p>

<b>Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects (2018)</b> [[Paper]](https://research.nvidia.com/publication/2018-09_Deep-Object-Pose)
<p align="center"><img width="50%" src="https://research.nvidia.com/sites/default/files/publications/forwebsite1_0.png" /></p>

<a name="single_classification" />

## Single Object Classification
:space_invader: <b>3D ShapeNets: A Deep Representation for Volumetric Shapes (2015)</b> [[Paper]](http://3dshapenets.cs.princeton.edu/)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/3ed23386284a5639cb3e8baaecf496caa766e335/1-Figure1-1.png" /></p>

:space_invader: <b>VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition (2015)</b> [[Paper]](http://www.dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf) [[Code]](https://github.com/dimatura/voxnet)
<p align="center"><img width="50%" src="http://www.dimatura.net/research/voxnet/car_voxnet_side.png" /></p>

:camera: <b>Multi-view Convolutional Neural Networks  for 3D Shape Recognition (2015)</b> [[Paper]](http://vis-www.cs.umass.edu/mvcnn/)
<p align="center"><img width="50%" src="http://vis-www.cs.umass.edu/mvcnn/images/mvcnn.png" /></p>

:camera: <b>DeepPano: Deep Panoramic Representation for 3-D Shape Recognition (2015)</b> [[Paper]](http://mclab.eic.hust.edu.cn/UpLoadFiles/Papers/DeepPano_SPL2015.pdf)
<p align="center"><img width="30%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/5a1b5d31905d8cece7b78510f51f3d8bbb063063/1-Figure3-1.png" /></p>

:space_invader::camera: <b>FusionNet: 3D Object Classification Using Multiple Data Representations (2016)</b> [[Paper]](https://stanford.edu/~rezab/papers/fusionnet.pdf)
<p align="center"><img width="30%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/0aab8fbcef1f0a14f5653d170ca36f4e5aae8010/6-Figure5-1.png" /></p>

:space_invader::camera: <b>Volumetric and Multi-View CNNs for Object Classification on 3D Data (2016)</b> [[Paper]](https://arxiv.org/pdf/1604.03265.pdf) [[Code]](https://github.com/charlesq34/3dcnn.torch)
<p align="center"><img width="40%" src="http://graphics.stanford.edu/projects/3dcnn/teaser.jpg" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2016)</b> [[Paper]](https://arxiv.org/pdf/1608.04236.pdf) [[Code]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="http://davidstutz.de/wordpress/wp-content/uploads/2017/02/brock_vae.png" /></p>

:gem: <b>Geometric deep learning on graphs and manifolds using mixture model CNNs (2016)</b> [[Link]](https://arxiv.org/pdf/1611.08402.pdf)
<p align="center"><img width="50%" src="https://i2.wp.com/preferredresearch.jp/wp-content/uploads/2017/08/monet.png?resize=581%2C155&ssl=1" /></p>

:space_invader: <b>3D GAN: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (2016)</b> [[Paper]](https://arxiv.org/pdf/1610.07584.pdf) [[Code]](https://github.com/zck119/3dgan-release)
<p align="center"><img width="50%" src="http://3dgan.csail.mit.edu/images/model.jpg" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2017)</b> [[Paper]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling/blob/master/doc/GUI3.png" /></p>

:space_invader: <b>FPNN: Field Probing Neural Networks for 3D Data (2016)</b> [[Paper]](http://yangyanli.github.io/FPNN/) [[Code]](https://github.com/yangyanli/FPNN)
<p align="center"><img width="30%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/15ca7adccf5cd4dc309cdcaa6328f4c429ead337/1-Figure2-1.png" /></p>

:space_invader: <b>OctNet: Learning Deep 3D Representations at High Resolutions (2017)</b> [[Paper]](https://arxiv.org/pdf/1611.05009.pdf) [[Code]](https://github.com/griegler/octnet)
<p align="center"><img width="30%" src="https://is.tuebingen.mpg.de/uploads/publication/image/18921/img03.png" /></p>

:space_invader: <b>O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis (2017)</b> [[Paper]](http://wang-ps.github.io/O-CNN) [[Code]](https://github.com/Microsoft/O-CNN)
<p align="center"><img width="50%" src="http://wang-ps.github.io/O-CNN_files/teaser.png" /></p>

:space_invader: <b>Orientation-boosted voxel nets for 3D object recognition (2017)</b> [[Paper]](https://lmb.informatik.uni-freiburg.de/Publications/2017/SZB17a/) [[Code]](https://github.com/lmb-freiburg/orion)
<p align="center"><img width="50%" src="https://lmb.informatik.uni-freiburg.de/Publications/2017/SZB17a/teaser_w.png" /></p>

:game_die: <b>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)</b> [[Paper]](http://stanford.edu/~rqi/pointnet/) [[Code]](https://github.com/charlesq34/pointnet)
<p align="center"><img width="40%" src="https://web.stanford.edu/~rqi/papers/pointnet.png" /></p>

:game_die: <b>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.02413.pdf) [[Code]](https://github.com/charlesq34/pointnet2)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointNet%2B%2B-%20Deep%20Hierarchical%20Feature%20Learning%20on%20Point%20Sets%20in%20a%20Metric%20Space.png" /></p>

:camera: <b>Feedback Networks (2017)</b> [[Paper]](http://feedbacknet.stanford.edu/) [[Code]](https://github.com/amir32002/feedback-networks)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Feedback%20Networks.png" /></p>

:game_die: <b>Escape from Cells: Deep Kd-Networks for The Recognition of 3D Point Cloud Models (2017)</b> [[Paper]](http://www.arxiv.org/pdf/1704.01222.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Escape From Cells.png" /></p>

:game_die: <b>Dynamic Graph CNN for Learning on Point Clouds (2018)</b> [[Paper]](https://arxiv.org/pdf/1801.07829.pdf)
<p align="center"><img width="50%" src="https://liuziwei7.github.io/homepage_files/dynamicgcnn_logo.png" /></p>

:game_die: <b>PointCNN (2018)</b> [[Paper]](https://yangyanli.github.io/PointCNN/)
<p align="center"><img width="50%" src="http://yangyan.li/images/paper/pointcnn.png" /></p>

:game_die::camera: <b>A Network Architecture for Point Cloud Classification via Automatic Depth Images Generation (2018 CVPR)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Roveri_A_Network_Architecture_CVPR_2018_paper.pdf)
<p align="center"><img width="50%" src="https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20180619114732/A-Network-Architecture-for-Point-Cloud-Classification-via-Automatic-Depth-Images-Generation-Image-600x317.jpg" /></p>

:game_die::space_invader: <b>PointGrid: A Deep Network for 3D Shape Understanding (CVPR 2018) </b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf) [[Code]](https://github.com/trucleduc/PointGrid)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointGrid-%20A%20Deep%20Network%20for%203D%20Shape%20Understanding%20(2018).jpeg" /></p>

:gem: <b> MeshNet: Mesh Neural Network for 3D Shape Representation (AAAI 2019) </b> [[Paper]](https://arxiv.org/pdf/1811.11424.pdf) [[Code]](https://github.com/Yue-Group/MeshNet)
<p align="center"><img width="50%" src="http://www.gaoyue.org/en_tsinghua/resrc/meshnet.jpg" /></p>

:game_die: <b>SpiderCNN (2018)</b> [[Paper]](https://github.com/xyf513/SpiderCNN)[[Code]](https://github.com/xyf513/SpiderCNN)
<p align="center"><img width="50%" src="http://5b0988e595225.cdn.sohucs.com/images/20181109/45c3b670e67f43b288791c650fb7fb0b.jpeg" /></p>

:game_die: <b>PointConv (2018)</b> [[Paper]](https://github.com/DylanWusee/pointconv/tree/master/imgs)[[Code]](https://github.com/DylanWusee/pointconv/tree/master/imgs)
<p align="center"><img width="50%" src="https://pics4.baidu.com/feed/8b82b9014a90f603272fe29f88ef061fb251ed49.jpeg?token=b23e1dbbaeaf12ffe3d168bd997a8d66&s=01307D328FE07C010C69C1CE0000D0B3" /></p>

<a name="multiple_detection" />


## Multiple Objects Detection
<b>Sliding Shapes for 3D Object Detection in Depth Images (2014)</b> [[Paper]](http://slidingshapes.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://slidingshapes.cs.princeton.edu/teaser.jpg" /></p>

<b>Object Detection in 3D Scenes Using CNNs in Multi-view Images (2016)</b> [[Paper]](https://stanford.edu/class/ee367/Winter2016/Qi_Report.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Object%20Detection%20in%203D%20Scenes%20Using%20CNNs%20in%20Multi-view%20Images.png" /></p>

<b>Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images (2016)</b> [[Paper]](http://dss.cs.princeton.edu/) [[Code]](https://github.com/shurans/DeepSlidingShape)
<p align="center"><img width="50%" src="http://3dvision.princeton.edu/slide/DSS.jpg" /></p>

<b>DeepContext: Context-Encoding Neural Pathways  for 3D Holistic Scene Understanding (2016)</b> [[Paper]](http://deepcontext.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://deepcontext.cs.princeton.edu/teaser.png" /></p>

<b>SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite (2017)</b> [[Paper]](http://rgbd.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://rgbd.cs.princeton.edu/teaser.jpg" /></p>

<b>VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection (2017)</b> [[Paper]](https://arxiv.org/pdf/1711.06396.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DPMtLhHXUAcQUj2.jpg" /></p>

<b>Frustum PointNets for 3D Object Detection from RGB-D Data (CVPR2018)</b> [[Paper]](https://arxiv.org/pdf/1711.08488.pdf)

<p align="center"><img width="50%" src="http://stanford.edu/~rqi/frustum-pointnets/images/teaser.jpg" /></p>

<b>A^2-Net: Molecular Structure Estimation from Cryo-EM Density Volumes (AAAI2019)</b> [[Paper]](https://arxiv.org/abs/1901.00785)

<p align="center"><img width="50%" src="imgs/a-square-net-min.jpg" /></p>

<b>Stereo R-CNN based 3D Object Detection for Autonomous Driving (CVPR2019)</b> [[Paper]](https://arxiv.org/abs/1902.09738v1)

<p align="center"><img width="50%" src="https://www.groundai.com/media/arxiv_projects/515338/system_newnew.png" /></p>



<a name="segmentation" />

## Scene/Object Semantic Segmentation
<b>Learning 3D Mesh Segmentation and Labeling (2010)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/LabelMeshes.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/0bf390e2a14f74bcc8838d5fb1c0c4cc60e92eb7/7-Figure7-1.png" /></p>

<b>Unsupervised Co-Segmentation of a Set of Shapes via Descriptor-Space Spectral Clustering (2011)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/sidi_siga11_coseg.pdf)
<p align="center"><img width="30%" src="http://people.scs.carleton.ca/~olivervankaick/cosegmentation/results6.png" /></p>

<b>Single-View Reconstruction via Joint Analysis of Image and Shape Collections (2015)</b> [[Paper]](https://www.cs.utexas.edu/~huangqx/modeling_sig15.pdf) [[Code]](https://github.com/huangqx/image_shape_align)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2015/05/single-view.png" /></p>

<b>3D Shape Segmentation with Projective Convolutional Networks (2017)</b> [[Paper]](http://people.cs.umass.edu/~kalo/papers/shapepfcn/) [[Code]](https://github.com/kalov/ShapePFCN)
<p align="center"><img width="50%" src="http://people.cs.umass.edu/~kalo/papers/shapepfcn/teaser.jpg" /></p>

<b>Learning Hierarchical Shape Segmentation and Labeling from Online Repositories (2017)</b> [[Paper]](http://cs.stanford.edu/~ericyi/project_page/hier_seg/index.html)
<p align="center"><img width="50%" src="http://cs.stanford.edu/~ericyi/project_page/hier_seg/figures/teaser.jpg" /></p>

:space_invader: <b>ScanNet (2017)</b> [[Paper]](https://arxiv.org/pdf/1702.04405.pdf) [[Code]](https://github.com/scannet/scannet)
<p align="center"><img width="50%" src="http://www.scan-net.org/img/voxel-predictions.jpg" /></p>

:game_die: <b>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)</b> [[Paper]](http://stanford.edu/~rqi/pointnet/) [[Code]](https://github.com/charlesq34/pointnet)
<p align="center"><img width="40%" src="https://web.stanford.edu/~rqi/papers/pointnet.png" /></p>

:game_die: <b>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.02413.pdf) [[Code]](https://github.com/charlesq34/pointnet2)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointNet%2B%2B-%20Deep%20Hierarchical%20Feature%20Learning%20on%20Point%20Sets%20in%20a%20Metric%20Space.png" /></p>

:game_die: <b>3D Graph Neural Networks for RGBD Semantic Segmentation (2017)</b> [[Paper]](http://www.cs.toronto.edu/~rjliao/papers/iccv_2017_3DGNN.pdf)
<p align="center"><img width="40%" src="http://www.fonow.com/Images/2017-10-18/66372-20171018115809740-2125227250.jpg" /></p>

:game_die: <b>3DCNN-DQN-RNN: A Deep Reinforcement Learning Framework for Semantic
Parsing of Large-scale 3D Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.06783.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/3DCNN-DQN-RNN.png" /></p>

:game_die::space_invader: <b>Semantic Segmentation of Indoor Point Clouds using Convolutional Neural Networks (2017)</b> [[Paper]](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-4-W4/101/2017/isprs-annals-IV-4-W4-101-2017.pdf)
<p align="center"><img width="55%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Semantic Segmentation of Indoor Point Clouds using Convolutional Neural Networks.png" /></p>

:game_die::space_invader: <b>SEGCloud: Semantic Segmentation of 3D Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.07563.pdf)
<p align="center"><img width="55%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/SEGCloud.png" /></p>

:game_die::space_invader: <b>Large-Scale 3D Shape Reconstruction and Segmentation from ShapeNet Core55 (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.06104.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Core55.png" /></p>

:game_die: <b>Dynamic Graph CNN for Learning on Point Clouds (2018)</b> [[Paper]](https://arxiv.org/pdf/1801.07829.pdf)
<p align="center"><img width="50%" src="https://liuziwei7.github.io/homepage_files/dynamicgcnn_logo.png" /></p>

:game_die: <b>PointCNN (2018)</b> [[Paper]](https://yangyanli.github.io/PointCNN/)
<p align="center"><img width="50%" src="http://yangyan.li/images/paper/pointcnn.png" /></p>

:camera::space_invader: <b>3DMV: Joint 3D-Multi-View Prediction for 3D Semantic Scene Segmentation (2018)</b> [[Paper]](https://arxiv.org/pdf/1803.10409.pdf)
<p align="center"><img width="50%" src="https://github.com/angeladai/3DMV/blob/master/images/teaser.jpg" /></p>

:space_invader: <b>ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans (2018)</b> [[Paper]](https://arxiv.org/pdf/1712.10215.pdf) 
<p align="center"><img width="50%" src="https://github.com/angeladai/ScanComplete/blob/master/images/teaser_mesh.jpg" /></p>

:game_die::camera: <b>SPLATNet: Sparse Lattice Networks for Point Cloud Processing (2018)</b> [[Paper]](https://arxiv.org/pdf/1802.08275.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/SPLATNet-%20Sparse%20Lattice%20Networks%20for%20Point%20Cloud%20Processing.jpeg" /></p>

:game_die::space_invader: <b>PointGrid: A Deep Network for 3D Shape Understanding (CVPR 2018) </b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf) [[Code]](https://github.com/trucleduc/PointGrid)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointGrid-%20A%20Deep%20Network%20for%203D%20Shape%20Understanding%20(2018).jpeg" /></p>

:game_die: <b>PointConv (2018)</b> [[Paper]](https://github.com/DylanWusee/pointconv/tree/master/imgs)[[Code]](https://github.com/DylanWusee/pointconv/tree/master/imgs)
<p align="center"><img width="50%" src="https://pics4.baidu.com/feed/8b82b9014a90f603272fe29f88ef061fb251ed49.jpeg?token=b23e1dbbaeaf12ffe3d168bd997a8d66&s=01307D328FE07C010C69C1CE0000D0B3" /></p>

:game_die: <b>SpiderCNN (2018)</b> [[Paper]](https://github.com/xyf513/SpiderCNN)[[Code]](https://github.com/xyf513/SpiderCNN)
<p align="center"><img width="50%" src="http://5b0988e595225.cdn.sohucs.com/images/20181109/45c3b670e67f43b288791c650fb7fb0b.jpeg" /></p>

<a name="3d_synthesis" />

## 3D Model Synthesis/Reconstruction

<a name="3d_synthesis_model_based" />

_Parametric Morphable Model-based methods_

<b>A Morphable Model For The Synthesis Of 3D Faces (1999)</b> [[Paper]](http://gravis.dmi.unibas.ch/publications/Sigg99/morphmod2.pdf)[[Code]](https://github.com/MichaelMure/3DMM)
<p align="center"><img width="40%" src="http://mblogthumb3.phinf.naver.net/MjAxNzAzMTdfMjcz/MDAxNDg5NzE3MzU0ODI3.9lQioLxwoGmtoIVXX9sbVOzhezoqgKMKiTovBnbUFN0g.sXN5tG4Kohgk7OJEtPnux-mv7OAoXVxxCyo3SGZMc6Yg.PNG.atelierjpro/031717_0222_DataDrivenS4.png?type=w420" /></p>

<b>The Space of Human Body Shapes: Reconstruction and Parameterization from Range Scans (2003)</b> [[Paper]](http://grail.cs.washington.edu/projects/digital-human/pub/allen03space-submit.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/46d39b0e21ae956e4bcb7a789f92be480d45ee12/7-Figure10-1.png" /></p>

<b>Category-Specific Object Reconstruction from a Single Image (2014)</b> [[Paper]](https://people.eecs.berkeley.edu/~akar/categoryshapes.pdf)
<p align="center"><img width="50%" src="http://people.eecs.berkeley.edu/~akar/categoryShapes/images/teaser.png" /></p>

:game_die: <b>DeformNet: Free-Form Deformation Network for 3D Shape Reconstruction from a Single Image (2017)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf)
<p align="center"><img width="50%" src="https://chrischoy.github.io/images/publication/deformnet/model.png" /></p>

:gem: <b>Mesh-based Autoencoders for Localized Deformation Component Analysis (2017)</b> [[Paper]](https://arxiv.org/pdf/1709.04304.pdf)
<p align="center"><img width="50%" src="http://qytan.com/img/point_conv.jpg" /></p>

:gem: <b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>

:gem: <b>Using Locally Corresponding CAD Models for
Dense 3D Reconstructions from a Single Image (2017)</b> [[Paper]](http://ci2cv.net/media/papers/chenkong_cvpr_2017.pdf)
<p align="center"><img width="50%" src="https://chenhsuanlin.bitbucket.io/images/rp/r02.png" /></p>

:gem: <b>Compact Model Representation for 3D Reconstruction (2017)</b> [[Paper]](https://jhonykaesemodel.com/publication/3dv2017/)
<p align="center"><img width="50%" src="https://jhonykaesemodel.com/img/headers/overview.png" /></p>

:gem: <b>Image2Mesh: A Learning Framework for Single Image 3D Reconstruction (2017)</b> [[Paper]](https://arxiv.org/pdf/1711.10669.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DW5VhjpW4AAESHO.jpg" /></p>

:gem: <b>Learning free-form deformations for 3D object reconstruction (2018)</b> [[Paper]](https://jhonykaesemodel.com/publication/learning_ffd/)
<p align="center"><img width="50%" src="https://jhonykaesemodel.com/learning_ffd_overview.png" /></p>

:gem: <b>Variational Autoencoders for Deforming 3D Mesh Models(2018 CVPR)</b> [[Paper]](http://qytan.com/publication/vae/)
<p align="center"><img width="50%" src="http://humanmotion.ict.ac.cn/papers/2018P5_VariationalAutoencoders/TeaserImage.jpg" /></p>

:gem: <b>Lions and Tigers and Bears: Capturing Non-Rigid, 3D, Articulated Shape from Images (2018 CVPR)</b> [[Paper]](http://files.is.tue.mpg.de/black/papers/zuffiCVPR2018.pdf)
<p align="center"><img width="50%" src="https://3c1703fe8d.site.internapcdn.net/newman/gfx/news/hires/2018/realisticava.jpg" /></p>

<a name="3d_synthesis_template_based" />

_Part-based Template Learning methods_

<b>Modeling by Example (2004)</b> [[Paper]](http://www.cs.princeton.edu/~funk/sig04a.pdf)
<p align="center"><img width="20%" src="http://gfx.cs.princeton.edu/pubs/Funkhouser_2004_MBE/chair.jpg" /></p>

<b>Model Composition from Interchangeable Components (2007)</b> [[Paper]](http://www.cs.princeton.edu/courses/archive/spring11/cos598A/pdfs/Kraevoy07.pdf)
<p align="center"><img width="40%" src="http://www.cs.ubc.ca/labs/imager/tr/2007/Vlad_Shuffler/teaser.jpg" /></p>

<b>Data-Driven Suggestions for Creativity Support in 3D Modeling (2010)</b> [[Paper]](http://vladlen.info/publications/data-driven-suggestions-for-creativity-support-in-3d-modeling/)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2011/12/creativity.png" /></p>

<b>Photo-Inspired Model-Driven 3D Object Modeling (2011)</b> [[Paper]](http://kevinkaixu.net/projects/photo-inspired.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/photo-inspired/overview.PNG" /></p>

<b>Probabilistic Reasoning for Assembly-Based 3D Modeling (2011)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/assembly/ProbReasoningShapeModeling.pdf)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2011/12/highlight9.png" /></p>

<b>A Probabilistic Model for Component-Based Shape Synthesis (2012)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/ShapeSynthesis/ShapeSynthesis.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/A%20Probabilistic%20Model%20for%20Component-Based%20Shape%20Synthesis.png" /></p>

<b>Structure Recovery by Part Assembly (2012)</b> [[Paper]](http://cg.cs.tsinghua.edu.cn/StructureRecovery/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/Structure%20Recovery%20by%20Part%20Assembly.png" /></p>

<b>Fit and Diverse: Set Evolution for Inspiring 3D Shape Galleries (2012)</b> [[Paper]](http://kevinkaixu.net/projects/civil.html)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/civil/teaser.png" /></p>

<b>AttribIt: Content Creation with Semantic Attributes (2013)</b> [[Paper]](https://people.cs.umass.edu/~kalo/papers/attribit/AttribIt.pdf)
<p align="center"><img width="30%" src="http://gfx.cs.princeton.edu/gfx/pubs/Chaudhuri_2013_ACC/teaser.jpg" /></p>

<b>Learning Part-based Templates from Large Collections of 3D Shapes (2013)</b> [[Paper]](http://shape.cs.princeton.edu/vkcorrs/papers/13_SIGGRAPH_CorrsTmplt.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/test1/blob/master/imgs/Learning%20Part-based%20Templates%20from%20Large%20Collections%20of%203D%20Shapes.png" /></p>

<b>Topology-Varying 3D Shape Creation via Structural Blending (2014)</b> [[Paper]](http://gruvi.cs.sfu.ca/project/topo/)
<p align="center"><img width="50%" src="https://i.ytimg.com/vi/Xc4qf7v6a-w/maxresdefault.jpg" /></p>

<b>Estimating Image Depth using Shape Collections (2014)</b> [[Paper]](http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/image_shape_net/imageShapeNet_sigg14.html)
<p align="center"><img width="50%" src="http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/image_shape_net/paper_docs/pipeline.jpg" /></p>

<b>Single-View Reconstruction via Joint Analysis of Image and Shape Collections (2015)</b> [[Paper]](https://www.cs.utexas.edu/~huangqx/modeling_sig15.pdf)
<p align="center"><img width="50%" src="http://vladlen.info/wp-content/uploads/2015/05/single-view.png" /></p>

<b>Interchangeable Components for Hands-On Assembly Based Modeling (2016)</b> [[Paper]](http://www.cs.umb.edu/~craigyu/papers/handson_low_res.pdf)
<p align="center"><img width="30%" src="https://github.com/timzhang642/test1/blob/master/imgs/Interchangeable%20Components%20for%20Hands-On%20Assembly%20Based%20Modeling.png" /></p>

<b>Shape Completion from a Single RGBD Image (2016)</b> [[Paper]](http://www.kunzhou.net/2016/shapecompletion-tvcg16.pdf)
<p align="center"><img width="40%" src="http://tianjiashao.com/Images/2015/completion.jpg" /></p>

<a name="3d_synthesis_dl_based" />

_Deep Learning Methods_

:camera: <b>Learning to Generate Chairs, Tables and Cars with Convolutional Networks (2014)</b> [[Paper]](https://arxiv.org/pdf/1411.5928.pdf)
<p align="center"><img width="50%" src="https://zo7.github.io/img/2016-09-25-generating-faces/chairs-model.png" /></p>

:camera: <b>Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis (2015, NIPS)</b> [[Paper]](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf)
<p align="center"><img width="50%" src="https://github.com/jimeiyang/deepRotator/blob/master/demo_img.png" /></p>

:game_die: <b>Analysis and synthesis of 3D shape families via deep-learned generative models of surfaces (2015)</b> [[Paper]](https://people.cs.umass.edu/~hbhuang/publications/bsm/)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~hbhuang/publications/bsm/bsm_teaser.jpg" /></p>

:camera: <b>Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis (2015)</b> [[Paper]](https://papers.nips.cc/paper/5639-weakly-supervised-disentangling-with-recurrent-transformations-for-3d-view-synthesis.pdf) [[Code]](https://github.com/jimeiyang/deepRotator)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/042993c46294a542946c9c1706b7b22deb1d7c43/2-Figure1-1.png" /></p>

:camera: <b>Multi-view 3D Models from Single Images with a Convolutional Network (2016)</b> [[Paper]](https://arxiv.org/pdf/1511.06702.pdf) [[Code]](https://github.com/lmb-freiburg/mv3d)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/3d7ca5ad34f23a5fab16e73e287d1a059dc7ef9a/4-Figure2-1.png" /></p>

:camera: <b>View Synthesis by Appearance Flow (2016)</b> [[Paper]](https://people.eecs.berkeley.edu/~tinghuiz/papers/eccv16_appflow.pdf) [[Code]](https://github.com/tinghuiz/appearance-flow)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/12280506dc8b5c3ca2db29fc3be694d9a8bef48c/6-Figure2-1.png" /></p>

:space_invader: <b>Voxlets: Structured Prediction of Unobserved Voxels From a Single Depth Image (2016)</b> [[Paper]](http://visual.cs.ucl.ac.uk/pubs/depthPrediction/http://visual.cs.ucl.ac.uk/pubs/depthPrediction/) [[Code]](https://github.com/mdfirman/voxlets)
<p align="center"><img width="30%" src="https://i.ytimg.com/vi/1wy4y2GWD5o/maxresdefault.jpg" /></p>

:space_invader: <b>3D-R2N2: 3D Recurrent Reconstruction Neural Network (2016)</b> [[Paper]](http://cvgl.stanford.edu/3d-r2n2/) [[Code]](https://github.com/chrischoy/3D-R2N2)
<p align="center"><img width="50%" src="http://3d-r2n2.stanford.edu/imgs/overview.png" /></p>

:space_invader: <b>Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision (2016)</b> [[Paper]](https://eng.ucmerced.edu/people/jyang44/papers/nips16_ptn.pdf)
<p align="center"><img width="70%" src="https://sites.google.com/site/skywalkeryxc/_/rsrc/1481104596238/perspective_transformer_nets/network_arch.png" /></p>

:space_invader: <b>TL-Embedding Network: Learning a Predictable and Generative Vector Representation for Objects (2016)</b> [[Paper]](https://arxiv.org/pdf/1603.08637.pdf)
<p align="center"><img width="50%" src="https://rohitgirdhar.github.io/GenerativePredictableVoxels/assets/webteaser.jpg" /></p>

:space_invader: <b>3D GAN: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (2016)</b> [[Paper]](https://arxiv.org/pdf/1610.07584.pdf)
<p align="center"><img width="50%" src="http://3dgan.csail.mit.edu/images/model.jpg" /></p>

:space_invader: <b>3D Shape Induction from 2D Views of Multiple Objects (2016)</b> [[Paper]](https://arxiv.org/pdf/1612.05872.pdf)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/e78572eeef8b967dec420013c65a6684487c13b2/2-Figure2-1.png" /></p>

:camera: <b>Unsupervised Learning of 3D Structure from Images (2016)</b> [[Paper]](https://arxiv.org/pdf/1607.00662.pdf)
<p align="center"><img width="50%" src="https://adriancolyer.files.wordpress.com/2016/12/unsupervised-3d-fig-10.jpeg?w=600" /></p>

:space_invader: <b>Generative and Discriminative Voxel Modeling with Convolutional Neural Networks (2016)</b> [[Paper]](https://arxiv.org/pdf/1608.04236.pdf) [[Code]](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
<p align="center"><img width="50%" src="http://davidstutz.de/wordpress/wp-content/uploads/2017/02/brock_vae.png" /></p>

:camera: <b>Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency (2017)</b> [[Paper]](https://shubhtuls.github.io/drc/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/drc/resources/images/teaserChair.png" /></p>

:camera: <b>Synthesizing 3D Shapes via Modeling Multi-View Depth Maps and Silhouettes with Deep Generative Networks (2017)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Soltani_Synthesizing_3D_Shapes_CVPR_2017_paper.pdf)  [[Code]](https://github.com/Amir-Arsalan/Synthesize3DviaDepthOrSil)
<p align="center"><img width="50%" src="https://jiajunwu.com/images/spotlight_3dvae.jpg" /></p>

:space_invader: <b>Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis (2017)</b> [[Paper]](https://arxiv.org/pdf/1612.00101.pdf) [[Code]](https://github.com/angeladai/cnncomplete)
<p align="center"><img width="50%" src="http://graphics.stanford.edu/projects/cnncomplete/teaser.jpg" /></p>

:space_invader: <b>Octree Generating Networks: Efficient Convolutional Architectures for High-resolution 3D Outputs (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.09438.pdf) [[Code]](https://github.com/lmb-freiburg/ogn)
<p align="center"><img width="50%" src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/6c2a292bb018a8742cbb0bbc5e23dd0a454ffe3a/2-Figure2-1.png" /></p>

:space_invader: <b>Hierarchical Surface Prediction for 3D Object Reconstruction (2017)</b> [[Paper]](https://arxiv.org/pdf/1704.00710.pdf)
<p align="center"><img width="50%" src="http://bair.berkeley.edu/blog/assets/hsp/image_2.png" /></p>

:space_invader: <b>OctNetFusion: Learning Depth Fusion from Data (2017)</b> [[Paper]](https://arxiv.org/pdf/1704.01047.pdf) [[Code]](https://github.com/griegler/octnetfusion)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/OctNetFusion-%20Learning%20Depth%20Fusion%20from%20Data.jpeg" /></p>

:game_die: <b>A Point Set Generation Network for 3D Object Reconstruction from a Single Image (2017)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf) [[Code]](https://github.com/fanhqme/PointSetGeneration)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/A%20Point%20Set%20Generation%20Network%20for%203D%20Object%20Reconstruction%20from%20a%20Single%20Image%20(2017).jpeg" /></p>

:game_die: <b>Learning Representations and Generative Models for 3D Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.02392.pdf) [[Code]](https://github.com/optas/latent_3d_points)
<p align="center"><img width="50%" src="https://github.com/optas/latent_3d_points/blob/master/doc/images/teaser.jpg" /></p>

:game_die: <b>Shape Generation using Spatially Partitioned Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.06267.pdf)
<p align="center"><img width="50%" src="http://mgadelha.me/sppc/fig/abstract.png" /></p>

:game_die: <b>PCPNET Learning Local Shape Properties from Raw Point Clouds (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.04954.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PCPNET%20Learning%20Local%20Shape%20Properties%20from%20Raw%20Point%20Clouds%20(2017).jpeg" /></p>

:camera: <b>Transformation-Grounded Image Generation Network for Novel 3D View Synthesis (2017)</b> [[Paper]](http://www.cs.unc.edu/~eunbyung/tvsn/) [[Code]](https://github.com/silverbottlep/tvsn)
<p align="center"><img width="50%" src="https://eng.ucmerced.edu/people/jyang44/pics/view_synthesis.gif" /></p>

:camera: <b>Tag Disentangled Generative Adversarial Networks for Object Image Re-rendering (2017)</b> [[Paper]](http://static.ijcai.org/proceedings-2017/0404.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Tag%20Disentangled%20Generative%20Adversarial%20Networks%20for%20Object%20Image%20Re-rendering.jpeg" /></p>

:camera: <b>3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks (2017)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/SketchModeling/) [[Code]](https://github.com/happylun/SketchModeling)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/SketchModeling/SketchModeling_teaser.png" /></p>

:space_invader: <b>Interactive 3D Modeling with a Generative Adversarial Network (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.05170.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DCsPKLqXoAEBd-V.jpg" /></p>

:camera::space_invader: <b>Weakly supervised 3D Reconstruction with Adversarial Constraint (2017)</b> [[Paper]](https://arxiv.org/pdf/1705.10904.pdf) [[Code]](https://github.com/jgwak/McRecon)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Weakly%20supervised%203D%20Reconstruction%20with%20Adversarial%20Constraint%20(2017).jpeg" /></p>

:camera: <b>SurfNet: Generating 3D shape surfaces using deep residual networks (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.04079.pdf)
<p align="center"><img width="50%" src="https://3dadept.com/wp-content/uploads/2017/07/Screenshot-from-2017-07-26-145521-e1501077539723.png" /></p>

:pill: <b>GRASS: Generative Recursive Autoencoders for Shape Structures (SIGGRAPH 2017)</b> [[Paper]](http://kevinkaixu.net/projects/grass.html) [[Code]](https://github.com/junli-lj/grass) [[code]](https://github.com/kevin-kaixu/grass_pytorch)
<p align="center"><img width="50%" src="http://kevinkaixu.net/projects/grass/teaser.jpg" /></p>

:pill: <b> 3D-PRNN: Generating Shape Primitives with Recurrent Neural Networks (2017)</b> [[Paper]](https://arxiv.org/pdf/1708.01648.pdf)[[code]](https://github.com/zouchuhang/3D-PRNN)
<p align="center"><img width="50%" src="https://github.com/zouchuhang/3D-PRNN/blob/master/figs/teasor.jpg" /></p>

:gem: <b>Neural 3D Mesh Renderer (2017)</b> [[Paper]](http://hiroharu-kato.com/projects_en/neural_renderer.html) [[Code]](https://github.com/hiroharu-kato/neural_renderer.git)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DPSm-4HWkAApEZd.jpg" /></p>

:game_die::space_invader: <b>Large-Scale 3D Shape Reconstruction and Segmentation from ShapeNet Core55 (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.06104.pdf)
<p align="center"><img width="40%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Core55.png" /></p>

:space_invader: <b>Pix2vox: Sketch-Based 3D Exploration with Stacked Generative Adversarial Networks (2017)</b> [[Code]](https://github.com/maxorange/pix2vox)
<p align="center"><img width="50%" src="https://github.com/maxorange/pix2vox/blob/master/img/sample.gif" /></p>

:camera::space_invader: <b>What You Sketch Is What You Get: 3D Sketching using Multi-View Deep Volumetric Prediction (2017)</b> [[Paper]](https://arxiv.org/pdf/1707.08390.pdf)
<p align="center"><img width="50%" src="https://arxiv-sanity-sanity-production.s3.amazonaws.com/render-output/31631/x1.png" /></p>

:camera::space_invader: <b>MarrNet: 3D Shape Reconstruction via 2.5D Sketches (2017)</b> [[Paper]](http://marrnet.csail.mit.edu/)
<p align="center"><img width="50%" src="http://marrnet.csail.mit.edu/images/model.jpg" /></p>

:camera::space_invader::game_die: <b>Learning a Multi-View Stereo Machine (2017 NIPS)</b> [[Paper]](http://bair.berkeley.edu/blog/2017/09/05/unified-3d/) 
<p align="center"><img width="50%" src="http://bair.berkeley.edu/static/blog/unified-3d/Network.png" /></p>

:space_invader: <b>3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions (2017)</b> [[Paper]](http://3dmatch.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://3dmatch.cs.princeton.edu/img/overview.jpg" /></p>

:space_invader: <b>Scaling CNNs for High Resolution Volumetric Reconstruction from a Single Image (2017)</b> [[Paper]](https://ieeexplore.ieee.org/document/8265323/)
<p align="center"><img width="50%" src="https://github.com/frankhjwx/3D-Machine-Learning/blob/master/imgs/Scaling%20CNN%20Reconstruction.png" /></p>

:pill: <b>ComplementMe: Weakly-Supervised Component Suggestions for 3D Modeling (2017)</b> [[Paper]](https://arxiv.org/pdf/1708.01841.pdf)
<p align="center"><img width="50%" src="https://mhsung.github.io/assets/images/complement-me/figure_2.png" /></p>

:game_die: <b>PU-Net: Point Cloud Upsampling Network (2018)</b> [[Paper]](https://arxiv.org/pdf/1801.06761.pdf) [[Code]](https://github.com/yulequan/PU-Net)
<p align="center"><img width="50%" src="http://appsrv.cse.cuhk.edu.hk/~lqyu/indexpics/Pu-Net.png" /></p> 

:camera::space_invader: <b>Multi-view Consistency as Supervisory Signal  for Learning Shape and Pose Prediction (2018 CVPR)</b> [[Paper]](https://shubhtuls.github.io/mvcSnP/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/mvcSnP/resources/images/teaser.png" /></p>

:camera::game_die: <b>Object-Centric Photometric Bundle Adjustment with Deep Shape Prior (2018)</b> [[Paper]](http://ci2cv.net/media/papers/WACV18.pdf)
<p align="center"><img width="50%" src="https://chenhsuanlin.bitbucket.io/images/rp/r06.png" /></p>

:camera::game_die: <b>Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction (2018 AAAI)</b> [[Paper]](https://chenhsuanlin.bitbucket.io/3D-point-cloud-generation/)
<p align="center"><img width="50%" src="https://chenhsuanlin.bitbucket.io/images/rp/r05.png" /></p>

:gem: <b>Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (2018)</b> [[Paper]](https://github.com/nywang16/Pixel2Mesh)
<p align="center"><img width="50%" src="https://www.groundai.com/media/arxiv_projects/188911/x2.png.750x0_q75_crop.png" /></p>

:gem: <b>AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation (2018 CVPR)</b> [[Paper]](http://imagine.enpc.fr/~groueixt/atlasnet/) [[Code]](https://github.com/ThibaultGROUEIX/AtlasNet)
<p align="center"><img width="50%" src="http://imagine.enpc.fr/~groueixt/atlasnet/imgs/teaser.small.png" /></p>

:space_invader::gem: <b>Deep Marching Cubes: Learning Explicit Surface Representations (2018 CVPR)</b> [[Paper]](http://www.cvlibs.net/publications/Liao2018CVPR.pdf)
<p align="center"><img width="50%" src="https://github.com/frankhjwx/3D-Machine-Learning/blob/master/imgs/Deep%20Marching%20Cubes.png" /></p>

:space_invader: <b>Im2Avatar: Colorful 3D Reconstruction from a Single Image (2018)</b> [[Paper]](https://arxiv.org/pdf/1804.06375v1.pdf)
<p align="center"><img width="50%" src="https://github.com/syb7573330/im2avatar/blob/master/misc/demo_teaser.png" /></p>

:gem: <b>Learning Category-Specific Mesh Reconstruction  from Image Collections (2018)</b> [[Paper]](https://akanazawa.github.io/cmr/#)
<p align="center"><img width="50%" src="https://akanazawa.github.io/cmr/resources/images/teaser.png" /></p>

:pill: <b>CSGNet: Neural Shape Parser for Constructive Solid Geometry (2018)</b> [[Paper]](https://arxiv.org/pdf/1712.08290.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DR-RgbaU8AEyjeW.jpg" /></p>

:space_invader: <b>Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings (2018)</b> [[Paper]](http://text2shape.stanford.edu/)
<p align="center"><img width="50%" src="http://text2shape.stanford.edu/figures/pull.png" /></p>

:space_invader::gem::camera: <b>Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation (2018)</b>  [[Paper]](https://arxiv.org/abs/1802.09987) [[Code]](https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation)
<p align="center"><img width="60%" src="imgs/decomposition_new.png" /> <img width="60%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Multi-View%20Silhouette%20and%20Depth%20Decomposition%20for%20High%20Resolution%203D%20Object%20Representation.png" /></p>

:space_invader::gem::camera: <b>Pixels, voxels, and views: A study of shape representations for single view 3D object shape prediction (2018 CVPR)</b>  [[Paper]](https://arxiv.org/abs/1804.06032)
<p align="center"><img width="60%" src="imgs/pixels-voxels-views-rgb2mesh.png" /> </p>

:camera::game_die: <b>Neural scene representation and rendering (2018)</b> [[Paper]](https://deepmind.com/blog/neural-scene-representation-and-rendering/)
<p align="center"><img width="50%" src="http://www.arimorcos.com/static/images/publication_images/gqn_image.png" /></p>

:pill: <b>Im2Struct: Recovering 3D Shape Structure from a Single RGB Image (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1804.05469.pdf)
<p align="center"><img width="50%" src="https://kevinkaixu.net/images/publications/niu_cvpr18.jpg" /></p>

:game_die: <b>FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1712.07262.pdf)
<p align="center"><img width="50%" src="http://simbaforrest.github.io/fig/FoldingNet.jpg" /></p>

:camera::space_invader: <b>Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling (2018 CVPR)</b> [[Paper]](http://pix3d.csail.mit.edu/)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Pix3D%20-%20Dataset%20and%20Methods%20for%20Single-Image%203D%20Shape%20Modeling%20(2018%20CVPR).png" /></p>

:gem: <b>3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare (2018 CVPR)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1128.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/3D-RCNN-%20Instance-level%203D%20Object%20Reconstruction%20via%20Render-and-Compare%20(2018%20CVPR).jpeg" /></p>

:space_invader: <b>Matryoshka Networks: Predicting 3D Geometry via Nested Shape Layers (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1804.10975.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Matryoshka%20Networks-%20Predicting%203D%20Geometry%20via%20Nested%20Shape%20Layers%20(2018%20CVPR).jpeg" /></p>

:space_invader: <b>Global-to-Local Generative Model for 3D Shapes (SIGGRAPH Asia 2018)</b> [[Paper]](http://vcc.szu.edu.cn/research/2018/G2L)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Global-to-Local%20Generative%20Model%20for%203D%20Shapes.jpg" /></p>

:gem::game_die::space_invader: <b>ALIGNet: Partial-Shape Agnostic Alignment via Unsupervised Learning (TOG 2018)</b> [[Paper]](https://bit.ly/alignet) [[Code]](https://github.com/ranahanocka/ALIGNet/)
<p align="center"><img width="50%" src="https://github.com/ranahanocka/ALIGNet/blob/master/docs/rep.png" /></p>

:game_die::space_invader: <b>PointGrid: A Deep Network for 3D Shape Understanding (CVPR 2018) </b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf) [[Code]](https://github.com/trucleduc/PointGrid)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/PointGrid-%20A%20Deep%20Network%20for%203D%20Shape%20Understanding%20(2018).jpeg" /></p>

:game_die: <b>GAL: Geometric Adversarial Loss for Single-View 3D-Object Reconstruction (2018)</b> [[Paper]](https://xjqi.github.io/GAL.pdf)
<p align="center"><img width="50%" src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-01237-3_49/MediaObjects/474213_1_En_49_Fig2_HTML.gif" /></p>

:game_die: <b>Visual Object Networks: Image Generation with Disentangled 3D Representation (2018)</b> [[Paper]](https://papers.nips.cc/paper/7297-visual-object-networks-image-generation-with-disentangled-3d-representations.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Visual%20Object%20Networks-%20Image%20Generation%20with%20Disentangled%203D%20Representation%20(2018).jpeg" /></p>

:space_invader: <b>Learning to Infer and Execute 3D Shape Programs (2019))</b> [[Paper]](http://shape2prog.csail.mit.edu/)
<p align="center"><img width="50%" src="http://shape2prog.csail.mit.edu/shape_files/teaser.jpg" /></p>

:space_invader: <b>Learning to Infer and Execute 3D Shape Programs (2019))</b> [[Paper]](https://arxiv.org/pdf/1901.05103.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DxFaW-mU8AEo9wc.jpg" /></p>

:gem: <b>Learning View Priors for Single-view 3D Reconstruction (CVPR 2019)</b> [[Paper]](http://hiroharu-kato.com/projects_en/view_prior_learning.html)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Learning%20View%20Priors%20for%20Single-view%203D%20Reconstruction.png" /></p>

<a name="material_synthesis" />

## Texture/Material Analysis and Synthesis
<b>Texture Synthesis Using Convolutional Neural Networks (2015)</b> [[Paper]](https://arxiv.org/pdf/1505.07376.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Texture%20Synthesis%20Using%20Convolutional%20Neural%20Networks.jpeg" /></p>

<b>Two-Shot SVBRDF Capture for Stationary Materials (SIGGRAPH 2015)</b> [[Paper]](https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/)
<p align="center"><img width="50%" src="https://mediatech.aalto.fi/publications/graphics/TwoShotSVBRDF/teaser.png" /></p>

<b>Reflectance Modeling by Neural Texture Synthesis (2016)</b> [[Paper]](https://mediatech.aalto.fi/publications/graphics/NeuralSVBRDF/)
<p align="center"><img width="50%" src="https://mediatech.aalto.fi/publications/graphics/NeuralSVBRDF/teaser.png" /></p>

<b>Modeling Surface Appearance from a Single Photograph using Self-augmented Convolutional Neural Networks (2017)</b> [[Paper]](http://msraig.info/~sanet/sanet.htm)
<p align="center"><img width="50%" src="http://msraig.info/~sanet/teaser.jpg" /></p>

<b>High-Resolution Multi-Scale Neural Texture Synthesis (2017)</b> [[Paper]](https://wxs.ca/research/multiscale-neural-synthesis/)
<p align="center"><img width="50%" src="https://wxs.ca/research/multiscale-neural-synthesis/multiscale-gram-marble.jpg" /></p>

<b>Reflectance and Natural Illumination from Single Material Specular Objects Using Deep Learning (2017)</b> [[Paper]](https://homes.cs.washington.edu/~krematas/Publications/reflectance-natural-illumination.pdf)
<p align="center"><img width="50%" src="http://www.vision.ee.ethz.ch/~georgous/images/tpami17_teaser2.png" /></p>

<b>Joint Material and Illumination Estimation from Photo Sets in the Wild (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.08313.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Joint%20Material%20and%20Illumination%20Estimation%20from%20Photo%20Sets%20in%20the%20Wild.jpeg" /></p>

<b>JWhat Is Around The Camera? (2017)</b> [[Paper]](https://arxiv.org/pdf/1611.09325v2.pdf)
<p align="center"><img width="50%" src="https://homes.cs.washington.edu/~krematas/my_images/arxiv16b_teaser.jpg" /></p>

<b>TextureGAN: Controlling Deep Image Synthesis with Texture Patches (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1706.02823.pdf)
<p align="center"><img width="50%" src="http://texturegan.eye.gatech.edu/img/paper_figure.png" /></p>

<b>Gaussian Material Synthesis (2018 SIGGRAPH)</b> [[Paper]](https://users.cg.tuwien.ac.at/zsolnai/gfx/gaussian-material-synthesis/)
<p align="center"><img width="50%" src="https://i.ytimg.com/vi/VM2ysCnD9GA/maxresdefault.jpg" /></p>

<b>Non-stationary Texture Synthesis by Adversarial Expansion (2018 SIGGRAPH)</b> [[Paper]](http://vcc.szu.edu.cn/research/2018/TexSyn)
<p align="center"><img width="50%" src="https://github.com/jessemelpolio/non-stationary_texture_syn/blob/master/imgs/teaser.png" /></p>

<b>Synthesized Texture Quality Assessment via Multi-scale Spatial and Statistical Texture Attributes of Image and Gradient Magnitude Coefficients (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1804.08020.pdf)
<p align="center"><img width="50%" src="https://user-images.githubusercontent.com/12434910/39275366-e18c7c1c-4899-11e8-8e61-05072618bbce.PNG" /></p>

<b>LIME: Live Intrinsic Material Estimation (2018 CVPR)</b> [[Paper]](https://gvv.mpi-inf.mpg.de/projects/LIME/)
<p align="center"><img width="50%" src="https://web.stanford.edu/~zollhoef/papers/CVPR18_Material/teaser.png" /></p>

<b>Single-Image SVBRDF Capture with a Rendering-Aware Deep Network (2018)</b> [[Paper]](https://team.inria.fr/graphdeco/fr/projects/deep-materials/)
<p align="center"><img width="50%" src="https://team.inria.fr/graphdeco/files/2018/08/teaser_v0.png" /></p>

<b>PhotoShape: Photorealistic Materials for Large-Scale Shape Collections (2018)</b> [[Paper]](https://keunhong.com/publications/photoshape/)
<p align="center"><img width="50%" src="https://keunhong.com/publications/photoshape/teaser.jpg" /></p>

<b>Learning Material-Aware Local Descriptors for 3D Shapes (2018)</b> [[Paper]](http://www.vovakim.com/papers/18_3DV_ShapeMatFeat.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Learning%20Material-Aware%20Local%20Descriptors%20for%203D%20Shapes%20(2018).jpeg" /></p>

<b>FrankenGAN: Guided Detail Synthesis for Building Mass Models 
using Style-Synchonized GANs (2018 SIGGRAPH Asia)</b> [[Paper]](http://geometry.cs.ucl.ac.uk/projects/2018/frankengan/)
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2018/frankengan/paper_docs/teaser.jpg" /></p>

<a name="style_transfer" />

## Style Learning and Transfer
<b>Style-Content Separation by Anisotropic Part Scales (2010)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/xu_siga10_style.pdf)
<p align="center"><img width="50%" src="https://sites.google.com/site/kevinkaixu/_/rsrc/1472852123106/publications/style_b.jpg?height=145&width=400" /></p>

<b>Design Preserving Garment Transfer (2012)</b> [[Paper]](https://hal.inria.fr/hal-00695903/file/GarmentTransfer.pdf)
<p align="center"><img width="30%" src="https://hal.inria.fr/hal-00695903v2/file/02_WomanToAll.jpg" /></p>

<b>Analogy-Driven 3D Style Transfer (2014)</b> [[Paper]](http://www.chongyangma.com/publications/st/index.html)
<p align="center"><img width="50%" src="http://www.chongyangma.com/publications/st/2014_st_teaser.png" /></p>

<b>Elements of Style: Learning Perceptual Shape Style Similarity (2015)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity/StyleSimilarity.pdf) [[Code]](https://github.com/happylun/StyleSimilarity)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/StyleSimilarity/StyleSimilarity_teaser.jpg" /></p>

<b>Functionality Preserving Shape Style Transfer (2016)</b> [[Paper]](http://people.cs.umass.edu/~zlun/papers/StyleTransfer/StyleTransfer.pdf) [[Code]](https://github.com/happylun/StyleTransfer)
<p align="center"><img width="50%" src="https://people.cs.umass.edu/~zlun/papers/StyleTransfer/StyleTransfer_teaser.jpg" /></p>

<b>Unsupervised Texture Transfer from Images to Model Collections (2016)</b> [[Paper]](http://ai.stanford.edu/~haosu/papers/siga16_texture_transfer_small.pdf)
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2016/texture_transfer/paper_docs/teaser.png" /></p>

<b>Learning Detail Transfer based on Geometric Features (2017)</b> [[Paper]](http://surfacedetails.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://surfacedetails.cs.princeton.edu/images/teaser.png" /></p>

<b>Co-Locating Style-Defining Elements on 3D Shapes (2017)</b> [[Paper]](http://people.scs.carleton.ca/~olivervankaick/pubs/style_elem.pdf)
<p align="center"><img width="50%" src="http://s2017.siggraph.org/sites/default/files/styles/large/public/images/events/c118-e100-publicimage_0-itok=yO8OegQO.png" /></p>

<b>Neural 3D Mesh Renderer (2017)</b> [[Paper]](http://hiroharu-kato.com/projects_en/neural_renderer.html) [[Code]](https://github.com/hiroharu-kato/neural_renderer.git)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DPSm-4HWkAApEZd.jpg" /></p>

<b>Appearance Modeling via Proxy-to-Image Alignment (2018)</b> [[Paper]](http://vcc.szu.edu.cn/research/2018/AppMod)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Appearance%20Modeling%20via%20Proxy-to-Image%20Alignment.png" /></p>

:gem: <b>Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (2018)</b> [[Paper]](http://bigvid.fudan.edu.cn/pixel2mesh/)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/DaIuEnfU0AAqesA.jpg" /></p>

<b>Automatic Unpaired Shape Deformation Transfer (SIGGRAPH Asia 2018)</b> [[Paper]](http://geometrylearning.com/ausdt/)
<p align="center"><img width="50%" src="http://geometrylearning.com/ausdt/imgs/teaser.png" /></p>

<a name="scene_synthesis" />

## Scene Synthesis/Reconstruction
<b>Make It Home: Automatic Optimization of Furniture Arrangement (2011, SIGGRAPH)</b> [[Paper]](http://people.sutd.edu.sg/~saikit/projects/furniture/index.html)
<p align="center"><img width="40%" src="https://www.cs.umb.edu/~craigyu/img/papers/furniture.gif" /></p>

<b>Interactive Furniture Layout Using Interior Design Guidelines (2011)</b> [[Paper]](http://graphics.stanford.edu/~pmerrell/furnitureLayout.htm)
<p align="center"><img width="50%" src="http://vis.berkeley.edu/papers/furnitureLayout/furnitureBig.jpg" /></p>

<b>Synthesizing Open Worlds with Constraints using Locally Annealed Reversible Jump MCMC (2012)</b> [[Paper]](http://graphics.stanford.edu/~lfyg/owl.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Synthesizing%20Open%20Worlds%20with%20Constraints%20using%20Locally%20Annealed%20Reversible%20Jump%20MCMC%20(2012).jpeg" /></p>

<b>Example-based Synthesis of 3D Object Arrangements (2012 SIGGRAPH Asia)</b> [[Paper]](http://graphics.stanford.edu/projects/scenesynth/)
<p align="center"><img width="60%" src="http://graphics.stanford.edu/projects/scenesynth/img/teaser.jpg" /></p>

<b>Sketch2Scene: Sketch-based Co-retrieval  and Co-placement of 3D Models  (2013)</b> [[Paper]](http://sweb.cityu.edu.hk/hongbofu/projects/sketch2scene_sig13/#.WWWge__ysb0)
<p align="center"><img width="40%" src="http://sunweilun.github.io/images/paper/sketch2scene_thumb.jpg" /></p>

<b>Action-Driven 3D Indoor Scene Evolution (2016)</b> [[Paper]](https://www.cs.sfu.ca/~haoz/pubs/ma_siga16_action.pdf)
<p align="center"><img width="50%" src="https://maruitx.github.io/project/adise/teaser.jpg" /></p>

<b>The Clutterpalette: An Interactive Tool for Detailing Indoor Scenes (2015)</b> [[Paper]](https://www.cs.umb.edu/~craigyu/papers/clutterpalette.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/The%20Clutterpalette-%20An%20Interactive%20Tool%20for%20Detailing%20Indoor%20Scenes.png" /></p>

<b>Relationship Templates for Creating Scene Variations (2016)</b> [[Paper]](http://geometry.cs.ucl.ac.uk/projects/2016/relationship-templates/)
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2016/relationship-templates/paper_docs/teaser.png" /></p>

<b>IM2CAD (2017)</b> [[Paper]](http://homes.cs.washington.edu/~izadinia/im2cad.html)
<p align="center"><img width="50%" src="http://i.imgur.com/KhtOeuB.jpg" /></p>

<b>Predicting Complete 3D Models of Indoor Scenes (2017)</b> [[Paper]](https://arxiv.org/pdf/1504.02437.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Predicting%20Complete%203D%20Models%20of%20Indoor%20Scenes.png" /></p>

<b>Complete 3D Scene Parsing from Single RGBD Image (2017)</b> [[Paper]](https://arxiv.org/pdf/1710.09490.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Complete%203D%20Scene%20Parsing%20from%20Single%20RGBD%20Image.jpeg" /></p>

<b>Raster-to-Vector: Revisiting Floorplan Transformation (2017, ICCV)</b> [[Paper]](http://www.cse.wustl.edu/~chenliu/floorplan-transformation.html) [[Code]](https://github.com/art-programmer/FloorplanTransformation)
<p align="center"><img width="50%" src="https://www.cse.wustl.edu/~chenliu/floorplan-transformation/teaser.png" /></p>

<b>Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes (2017)</b> [[Blog]](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)
<p align="center"><img width="50%" src="https://cdn-images-1.medium.com/max/1600/1*NckW2hfgbHhEP3P8Z5ZLjQ.png" /></p>

<b>Adaptive Synthesis of Indoor Scenes via Activity-Associated Object Relation Graphs (2017 SIGGRAPH Asia)</b> [[Paper]](http://arts.buaa.edu.cn/projects/sa17/)
<p align="center"><img width="50%" src="https://sa2017.siggraph.org/images/events/c121-e45-publicimage.jpg" /></p>

<b>Automated Interior Design Using a Genetic Algorithm (2017)</b> [[Paper]](https://publik.tuwien.ac.at/files/publik_262718.pdf)
<p align="center"><img width="50%" src="http://www.peterkan.com/pictures/teaserq.jpg" /></p>

<b>SceneSuggest: Context-driven 3D Scene Design (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.00061.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/SceneSuggest%20-Context-driven%203D%20Scene%20Design%20(2017).png" /></p>

<b>A fully end-to-end deep learning approach for real-time simultaneous 3D reconstruction and material recognition (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.04699v1.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/A%20fully%20end-to-end%20deep%20learning%20approach%20for%20real-time%20simultaneous%203D%20reconstruction%20and%20material%20recognition%20(2017).png" /></p>

<b>Human-centric Indoor Scene Synthesis Using Stochastic Grammar (2018, CVPR)</b>[[Paper]](http://web.cs.ucla.edu/~syqi/publications/cvpr2018synthesis/cvpr2018synthesis.pdf) [[Supplementary]](http://web.cs.ucla.edu/~syqi/publications/cvpr2018synthesis/cvpr2018synthesis_supplementary.pdf) [[Code]](https://github.com/SiyuanQi/human-centric-scene-synthesis)
<p align="center"><img width="50%" src="http://web.cs.ucla.edu/~syqi/publications/thumbnails/cvpr2018synthesis.gif" /></p>

:camera::game_die: <b>FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans (2018)</b> [[Paper]](https://arxiv.org/pdf/1804.00090.pdf) [[Code]](http://art-programmer.github.io/floornet.html)
<p align="center"><img width="50%" src="http://art-programmer.github.io/floornet/teaser.png" /></p>

:space_invader: <b>ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans (2018)</b> [[Paper]](https://arxiv.org/pdf/1712.10215.pdf) 
<p align="center"><img width="50%" src="https://niessnerlab.org/papers/2018/3scancomplete/teaser.jpg" /></p>

<b>Deep Convolutional Priors for Indoor Scene Synthesis (2018)</b> [[Paper]](https://kwang-ether.github.io/pdf/deepsynth.pdf) 
<p align="center"><img width="50%" src="http://msavva.github.io/files/deepsynth.png" /></p>

<b>Configurable 3D Scene Synthesis and 2D Image Rendering
with Per-Pixel Ground Truth using Stochastic Grammars (2018)</b> [[Paper]](https://arxiv.org/pdf/1704.00112.pdf) 
<p align="center"><img width="50%" src="https://media.springernature.com/original/springer-static/image/art%3A10.1007%2Fs11263-018-1103-5/MediaObjects/11263_2018_1103_Fig5_HTML.jpg" /></p>

<b>Holistic 3D Scene Parsing and Reconstruction from a Single RGB Image (ECCV 2018)</b> [[Paper]](http://siyuanhuang.com/holistic_parsing/main.html) 
<p align="center"><img width="50%" src="http://web.cs.ucla.edu/~syqi/publications/thumbnails/eccv2018scene.png" /></p>

<b>Language-Driven Synthesis of 3D Scenes from Scene Databases (SIGGRAPH Asia 2018)</b> [[Paper]](http://www.sfu.ca/~agadipat/publications/2018/T2S/project_page.html) 
<p align="center"><img width="50%" src="http://www.sfu.ca/~agadipat/publications/2018/T2S/teaser.png" /></p>

<b>Deep Generative Modeling for Scene Synthesis via Hybrid Representations (2018)</b> [[Paper]](https://arxiv.org/pdf/1808.02084.pdf) 
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Deep%20Generative%20Modeling%20for%20Scene%20Synthesis%20via%20Hybrid%20Representations%20(2018).jpeg" /></p>

<b>GRAINS: Generative Recursive Autoencoders for INdoor Scenes (2018)</b> [[Paper]](https://arxiv.org/pdf/1807.09193.pdf) 
<p align="center"><img width="50%" src="https://www.groundai.com/media/arxiv_projects/373503/new_pics/teaserfig.jpg.750x0_q75_crop.jpg" /></p>

<b>SEETHROUGH: Finding Objects in Heavily Occluded Indoor Scene Images (2018)</b> [[Paper]](http://www.vovakim.com/papers/18_3DVOral_SeeThrough.pdf) 
<p align="center"><img width="50%" src="http://geometry.cs.ucl.ac.uk/projects/2018/seethrough/paper_docs/result_plate.png" /></p>

<a name="scene_understanding" />

## Scene Understanding
<b>Recovering the Spatial Layout of Cluttered Rooms (2009)</b> [[Paper]](http://dhoiem.cs.illinois.edu/publications/iccv2009_hedau_indoor.pdf)
<p align="center"><img width="60%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Recovering%20the%20Spatial%20Layout%20of%20Cluttered%20Rooms.png" /></p>

<b>Characterizing Structural Relationships in Scenes Using Graph Kernels (2011 SIGGRAPH)</b> [[Paper]](https://graphics.stanford.edu/~mdfisher/graphKernel.html)
<p align="center"><img width="60%" src="https://graphics.stanford.edu/~mdfisher/papers/graphKernelTeaser.png" /></p>

<b>Understanding Indoor Scenes Using 3D Geometric Phrases (2013)</b> [[Paper]](http://cvgl.stanford.edu/projects/3dgp/)
<p align="center"><img width="30%" src="http://cvgl.stanford.edu/projects/3dgp/images/title.png" /></p>

<b>Organizing Heterogeneous Scene Collections through Contextual Focal Points (2014 SIGGRAPH)</b> [[Paper]](http://kevinkaixu.net/projects/focal.html)
<p align="center"><img width="60%" src="http://kevinkaixu.net/projects/focal/overlapping_clusters.jpg" /></p>

<b>SceneGrok: Inferring Action Maps in 3D Environments (2014, SIGGRAPH)</b> [[Paper]](http://graphics.stanford.edu/projects/scenegrok/)
<p align="center"><img width="50%" src="http://graphics.stanford.edu/projects/scenegrok/scenegrok.png" /></p>

<b>PanoContext: A Whole-room 3D Context Model for Panoramic Scene Understanding (2014)</b> [[Paper]](http://panocontext.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://panocontext.cs.princeton.edu/teaser.jpg" /></p>

<b>Learning Informative Edge Maps for Indoor Scene Layout Prediction (2015)</b> [[Paper]](http://web.engr.illinois.edu/~slazebni/publications/iccv15_informative.pdf)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Learning%20Informative%20Edge%20Maps%20for%20Indoor%20Scene%20Layout%20Prediction.png" /></p>

<b>Rent3D: Floor-Plan Priors for Monocular Layout Estimation (2015)</b> [[Paper]](http://www.cs.toronto.edu/~fidler/projects/rent3D.html)
<p align="center"><img width="50%" src="http://www.cs.toronto.edu/~fidler/projects/layout-res.jpg" /></p>

<b>A Coarse-to-Fine Indoor Layout Estimation (CFILE) Method (2016)</b> [[Paper]](https://pdfs.semanticscholar.org/7024/a92186b81e6133dc779f497d06877b48d82b.pdf?_ga=2.54181869.497995160.1510977308-665742395.1510465328)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/A%20Coarse-to-Fine%20Indoor%20Layout%20Estimation%20(CFILE)%20Method%20(2016).png" /></p>

<b>DeLay: Robust Spatial Layout Estimation for Cluttered Indoor Scenes (2016)</b> [[Paper]](http://deeplayout.stanford.edu/)
<p align="center"><img width="30%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/DeLay-Robust%20Spatial%20Layout%20Estimation%20for%20Cluttered%20Indoor%20Scenes.png" /></p>

<b>3D Semantic Parsing of Large-Scale Indoor Spaces (2016)</b> [[Paper]](http://buildingparser.stanford.edu/method.html) [[Code]](https://github.com/alexsax/2D-3D-Semantics)
<p align="center"><img width="50%" src="http://buildingparser.stanford.edu/images/teaser.png" /></p>

<b>Single Image 3D Interpreter Network (2016)</b> [[Paper]](http://3dinterpreter.csail.mit.edu/) [[Code]](https://github.com/jiajunwu/3dinn)
<p align="center"><img width="50%" src="http://3dinterpreter.csail.mit.edu/images/spotlight_3dinn_large.jpg" /></p>

<b>Deep Multi-Modal Image Correspondence Learning (2016)</b> [[Paper]](http://www.cse.wustl.edu/~chenliu/floorplan-matching.html)
<p align="center"><img width="50%" src="http://art-programmer.github.io/floorplan-matching/teaser.png" /></p>

<b>Physically-Based Rendering for Indoor Scene Understanding Using Convolutional Neural Networks (2017)</b> [[Paper]](http://3dvision.princeton.edu/projects/2016/PBRS/) [[Code]](https://github.com/yindaz/pbrs) [[Code]](https://github.com/yindaz/surface_normal) [[Code]](https://github.com/fyu/dilation) [[Code]](https://github.com/s9xie/hed)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/C0YERJOXEAA69xN.jpg" /></p>

<b>RoomNet: End-to-End Room Layout Estimation (2017)</b> [[Paper]](https://arxiv.org/pdf/1703.06241.pdf)
<p align="center"><img width="50%" src="https://pbs.twimg.com/media/C7Z29GsV0AASEvR.jpg" /></p>

<b>SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite (2017)</b> [[Paper]](http://rgbd.cs.princeton.edu/)
<p align="center"><img width="50%" src="http://rgbd.cs.princeton.edu/teaser.jpg" /></p>

<b>Semantic Scene Completion from a Single Depth Image (2017)</b> [[Paper]](http://sscnet.cs.princeton.edu/) [[Code]](https://github.com/shurans/sscnet)
<p align="center"><img width="50%" src="http://sscnet.cs.princeton.edu/teaser.jpg" /></p>

<b>Factoring Shape, Pose, and Layout  from the 2D Image of a 3D Scene (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1712.01812.pdf) [[Code]](https://shubhtuls.github.io/factored3d/)
<p align="center"><img width="50%" src="https://shubhtuls.github.io/factored3d/resources/images/teaser.png" /></p>

<b>LayoutNet: Reconstructing the 3D Room Layout from a Single RGB Image (2018 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1803.08999.pdf) [[Code]](https://github.com/zouchuhang/LayoutNet)
<p align="center"><img width="50%" src="http://p0.ifengimg.com/pmop/2018/0404/A1D0CAE48130C918FE624FA60495F237C67172F6_size63_w797_h755.jpeg" /></p>

<b>PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image (2018 CVPR)</b> [[Paper]](http://art-programmer.github.io/planenet/paper.pdf) [[Code]](http://art-programmer.github.io/planenet.html)
<p align="center"><img width="50%" src="http://art-programmer.github.io/images/planenet.png" /></p>

<b>Cross-Domain Self-supervised Multi-task Feature Learning using Synthetic Imagery (2018 CVPR)</b> [[Paper]](http://web.cs.ucdavis.edu/~yjlee/projects/cvpr2018.pdf) <p align="center"><img width="50%" src="https://jason718.github.io/project/cvpr18/files/concept_pic.png" /></p>

<b>Pano2CAD: Room Layout From A Single Panorama Image (2018 CVPR)</b> [[Paper]](http://bjornstenger.github.io/papers/xu_wacv2017.pdf) <p align="center"><img width="50%" src="https://www.groundai.com/media/arxiv_projects/58924/figures/Compare_2b.png" /></p>

<b>Automatic 3D Indoor Scene Modeling from Single Panorama (2018 CVPR)</b> [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Automatic_3D_Indoor_CVPR_2018_paper.pdf) <p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Automatic%203D%20Indoor%20Scene%20Modeling%20from%20Single%20Panorama%20(2018%20CVPR).jpeg" /></p>

<b>Single-Image Piece-wise Planar 3D Reconstruction via Associative Embedding (2019 CVPR)</b> [[Paper]](https://arxiv.org/pdf/1902.09777.pdf) [[Code]](https://github.com/svip-lab/PlanarReconstruction) <p align="center"><img width="50%" src="https://github.com/svip-lab/PlanarReconstruction/blob/master/misc/pipeline.jpg" /></p>

<b>3D-Aware Scene Manipulation via Inverse Graphics (NeurIPS 2018)</b> [[Paper]](http://3dsdn.csail.mit.edu/) [[Code]](https://github.com/svip-lab/PlanarReconstruction) <p align="center"><img width="50%" src="http://3dsdn.csail.mit.edu/images/teaser.png" /></p>



# awesome-point-cloud-analysis [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
## forked from https://github.com/Yochengliu/awesome-point-cloud-analysis
## It's also a great document for 3D deep learning.

for anyone who wants to do research about 3D point cloud.   

If you find the awesome paper/code/dataset or have some suggestions, please contact linhua2017@ia.ac.cn. Thanks for your valuable contribution to the research community :smiley:   

<h1> 

```diff
- Recent papers (from 2017)
```

</h1>

<h3> Keywords </h3>

__`dat.`__: dataset &emsp; | &emsp; __`cls.`__: classification &emsp; | &emsp; __`rel.`__: retrieval &emsp; | &emsp; __`seg.`__: segmentation     
__`det.`__: detection &emsp; | &emsp; __`tra.`__: tracking &emsp; | &emsp; __`pos.`__: pose &emsp; | &emsp; __`dep.`__: depth     
__`reg.`__: registration &emsp; | &emsp; __`rec.`__: reconstruction &emsp; | &emsp; __`aut.`__: autonomous driving     
__`oth.`__: other, including normal-related, correspondence, mapping, matching, alignment, compression, generative model...

Statistics: :fire: code is available & stars >= 100 &emsp;|&emsp; :star: citation >= 50

---
## 2017
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [[tensorflow](https://github.com/charlesq34/pointnet)][[pytorch](https://github.com/fxia22/pointnet.pytorch)] [__`cls.`__ __`seg.`__ __`det.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.pdf)] Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs. [__`cls.`__] :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yi_SyncSpecCNN_Synchronized_Spectral_CVPR_2017_paper.pdf)] SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation. [[torch](https://github.com/ericyi/SyncSpecCNN)] [__`seg.`__ __`oth.`__] :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Dai_ScanNet_Richly-Annotated_3D_CVPR_2017_paper.pdf)] ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. [[project](http://www.scan-net.org/)][[git](http://www.scan-net.org/)] [__`dat.`__ __`cls.`__ __`rel.`__ __`seg.`__ __`oth.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mostegel_Scalable_Surface_Reconstruction_CVPR_2017_paper.pdf)] Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity. [__`oth.`__] 
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Straub_Efficient_Global_Point_CVPR_2017_paper.pdf)] Efficient Global Point Cloud Alignment using Bayesian Nonparametric Mixtures. [[code]( http://people.csail.mit.edu/jstraub/)] [__`oth.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Vongkulbhisal_Discriminative_Optimization_Theory_CVPR_2017_paper.pdf)] Discriminative Optimization: Theory and Applications to Point Cloud Registration. [__`reg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Elbaz_3D_Point_Cloud_CVPR_2017_paper.pdf)] 3D Point Cloud Registration for Localization using a Deep Neural Network Auto-Encoder. [[git](https://github.com/gilbaz/LORAX)] [__`reg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)] Multi-View 3D Object Detection Network for Autonomous Driving. [[tensorflow](https://github.com/bostondiditeam/MV3D)] [__`det.`__ __`aut.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)] 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions. [[code](https://github.com/andyzeng/3dmatch-toolbox)] [__`dat.`__ __`pos.`__ __`reg.`__ __`rec.`__ __`oth.`__] :fire: :star:
-
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Klokov_Escape_From_Cells_ICCV_2017_paper.pdf)] Escape from Cells: Deep Kd-Networks for the Recognition of 3D Point Cloud Models. [[pytorch](https://github.com/fxia22/kdnet.pytorch)] [__`cls.`__ __`rel.`__ __`seg.`__] :star:
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_3DCNN-DQN-RNN_A_Deep_ICCV_2017_paper.pdf)] 3DCNN-DQN-RNN: A Deep Reinforcement Learning Framework for Semantic Parsing of Large-scale 3D Point Clouds. [[code](https://github.com/CKchaos/scn2pointcloud_tool)] [__`seg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)] Colored Point Cloud Registration Revisited. [__`reg.`__]
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Nan_PolyFit_Polygonal_Surface_ICCV_2017_paper.pdf)] PolyFit: Polygonal Surface Reconstruction from Point Clouds. [[code](https://github.com/LiangliangNan/PolyFit)] [__`rec.`__] :fire:
- [[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ladicky_From_Point_Clouds_ICCV_2017_paper.pdf)] From Point Clouds to Mesh using Regression. [__`rec.`__]
- 
- [[NeurIPS](https://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space)] PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. [[tensorflow](https://github.com/charlesq34/pointnet2)][[pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch)] [__`cls.`__ __`seg.`__] :fire: :star:
- [[NeurIPS](https://papers.nips.cc/paper/6931-deep-sets)] Deep Sets. [[pytorch](https://github.com/manzilzaheer/DeepSets)] [__`cls.`__] :star:
- 
- [[ICRA](https://ieeexplore.ieee.org/document/7989161)] Vote3Deep: Fast object detection in 3D point clouds using efficient convolutional neural networks. [[code](https://github.com/lijiannuist/Vote3Deep_lidar)] [__`det.`__ __`aut.`__] :star:
- [[ICRA](https://ieeexplore.ieee.org/document/7989591)] Fast segmentation of 3D point clouds: A paradigm on LiDAR data for autonomous vehicle applications. [[code](https://github.com/VincentCheungM/Run_based_segmentation)] [__`seg.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/document/7989618)] SegMatch: Segment based place recognition in 3D point clouds. [__`seg.`__ __`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/document/7989664)] Using 2 point+normal sets for fast registration of point clouds with small overlap. [__`reg.`__]
- 
- [[IROS](https://ieeexplore.ieee.org/document/8202234)] Car detection for autonomous vehicle: LIDAR and vision fusion approach through deep learning framework. [__`det.`__ __`aut.`__]
- [[IROS](https://ieeexplore.ieee.org/document/8202239)] 3D object classification with point convolution network. [__`cls.`__]
- [[IROS](https://ieeexplore.ieee.org/document/8205955)] 3D fully convolutional network for vehicle detection in point cloud. [[tensorflow](https://github.com/yukitsuji/3D_CNN_tensorflow)] [__`det.`__ __`aut.`__] :fire: :star:
- [[IROS](https://ieeexplore.ieee.org/document/8206488)] Deep learning of directional truncated signed distance function for robust 3D object recognition. [__`det.`__ __`pos.`__]
- [[IROS](https://ieeexplore.ieee.org/document/8206584)] Analyzing the quality of matched 3D point clouds of objects. [__`oth.`__]
-
- [[TPAMI](https://ieeexplore.ieee.org/ielx7/34/8454009/08046026.pdf?tp=&arnumber=8046026&isnumber=8454009&ref=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8=)] Structure-aware Data Consolidation. [__`oth.`__]

---
## 2018
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Su_SPLATNet_Sparse_Lattice_CVPR_2018_paper.pdf)] SPLATNet: Sparse Lattice Networks for Point Cloud Processing. [[caffe](https://github.com/NVlabs/splatnet)] [__`seg.`__] :fire:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.pdf)] Attentional ShapeContextNet for Point Cloud Recognition. [__`cls.`__ __`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Mining_Point_Cloud_CVPR_2018_paper.pdf)] Mining Point Cloud Local Structures by Kernel Correlation and Graph Pooling. [[code](http://www.merl.com/research/license#KCNet)] [__`cls.`__ __`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_FoldingNet_Point_Cloud_CVPR_2018_paper.pdf)] FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation. [[code](http://www.merl.com/research/license#FoldingNet)] [__`cls.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hua_Pointwise_Convolutional_Neural_CVPR_2018_paper.pdf)] Pointwise Convolutional Neural Networks. [[tensorflow](https://github.com/scenenn/pointwise)] [__`cls.`__ __`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_PU-Net_Point_Cloud_CVPR_2018_paper.pdf)] PU-Net: Point Cloud Upsampling Network. [[tensorflow](https://github.com/yulequan/PU-Net)] [__`rec.`__ __`oth.`__] :fire:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_SO-Net_Self-Organizing_Network_CVPR_2018_paper.pdf)] SO-Net: Self-Organizing Network for Point Cloud Analysis. [[pytorch](https://github.com/lijx10/SO-Net)] [__`cls.`__ __`seg.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Recurrent_Slice_Networks_CVPR_2018_paper.pdf)] Recurrent Slice Networks for 3D Segmentation of Point Clouds. [[pytorch](https://github.com/qianguih/RSNet)] [__`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Graham_3D_Semantic_Segmentation_CVPR_2018_paper.pdf)] 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks. [[pytorch](https://github.com/facebookresearch/SparseConvNet)] [__`seg.`__] :fire:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf)] Deep Parametric Continuous Convolutional Neural Networks. [__`seg.`__ __`aut.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf)] PIXOR: Real-time 3D Object Detection from Point Clouds. [[pytorch](https://github.com/ankita-kalra/PIXOR)] [__`det.`__ __`aut.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SGPN_Similarity_Group_CVPR_2018_paper.pdf)] SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation. [[tensorflow](https://github.com/laughtervv/SGPN)] [__`seg.`__] :fire:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf)] Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs. [[pytorch](https://github.com/loicland/superpoint_graph)] [__`seg.`__] :fire:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf)] VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. [[tensorflow](https://github.com/tsinghua-rll/VoxelNet-tensorflow)] [__`det.`__ __`aut.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yun_Reflection_Removal_for_CVPR_2018_paper.pdf)] Reflection Removal for Large-Scale 3D Point Clouds. [__`oth.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ge_Hand_PointNet_3D_CVPR_2018_paper.pdf)] Hand PointNet: 3D Hand Pose Estimation using Point Sets. [[pytorch](https://github.com/3huo/Hand-Pointnet)] [__`pos.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Uy_PointNetVLAD_Deep_Point_CVPR_2018_paper.pdf)] PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition. [[tensorflow](https://github.com/mikacuy/pointnetvlad.git)] [__`rel.`__] :fire:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Roveri_A_Network_Architecture_CVPR_2018_paper.pdf)] A Network Architecture for Point Cloud Classification via Automatic Depth Images Generation. [__`cls.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)] Density Adaptive Point Set Registration. [[code](https://github.com/felja633/DARE)] [__`reg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Birdal_A_Minimalist_Approach_CVPR_2018_paper.pdf)] A Minimalist Approach to Type-Agnostic Detection of Quadrics in Point Clouds. [__`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Vongkulbhisal_Inverse_Composition_Discriminative_CVPR_2018_paper.pdf)] Inverse Composition Discriminative Optimization for Point Cloud Registration. [__`reg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Reddy_CarFusion_Combining_Point_CVPR_2018_paper.pdf)] CarFusion: Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles. [__`tra.`__ __`det.`__ __`rec.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_PPFNet_Global_Context_CVPR_2018_paper.pdf)] PPFNet: Global Context Aware Local Features for Robust 3D Point Matching. [__`oth.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf)] PointGrid: A Deep Network for 3D Shape Understanding. [[tensorflow](https://github.com/trucleduc/PointGrid)] [__`cls.`__ __`seg.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_PointFusion_Deep_Sensor_CVPR_2018_paper.pdf)] PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation. [[code](https://github.com/malavikabindhi/CS230-PointFusion)] [__`det.`__ __`aut.`__]
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)] Frustum PointNets for 3D Object Detection from RGB-D Data. [[tensorflow](https://github.com/charlesq34/frustum-pointnets)] [__`det.`__ __`aut.`__] :fire: :star:
- [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tatarchenko_Tangent_Convolutions_for_CVPR_2018_paper.pdf)] Tangent Convolutions for Dense Prediction in 3D. [[tensorflow](https://github.com/tatarchm/tangent_conv)] [__`seg.`__ __`aut.`__]
- 
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Matheus_Gadelha_Multiresolution_Tree_Networks_ECCV_2018_paper.pdf)] Multiresolution Tree Networks for 3D Point Cloud Processing. [[pytorch](https://github.com/matheusgadelha/MRTNet)] [__`cls.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lequan_Yu_EC-Net_an_Edge-aware_ECCV_2018_paper.pdf)] EC-Net: an Edge-aware Point set Consolidation Network. [[tensorflow](https://github.com/yulequan/EC-Net)] [__`oth.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoqing_Ye_3D_Recurrent_Neural_ECCV_2018_paper.pdf)] 3D Recurrent Neural Networks with Context Fusion for Point Cloud Semantic Segmentation. [__`seg.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhou_Learning_and_Matching_ECCV_2018_paper.pdf)] Learning and Matching Multi-View Descriptors for Registration of Point Clouds. [__`reg.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. [[tensorflow](https://github.com/yewzijian/3DFeatNet)] [__`reg.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chu_Wang_Local_Spectral_Graph_ECCV_2018_paper.pdf)] Local Spectral Graph Convolution for Point Set Feature Learning. [[tensorflow](https://github.com/fate3439/LocalSpecGCN)] [__`cls.`__ __`seg.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Xu_SpiderCNN_Deep_Learning_ECCV_2018_paper.pdf)] SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters. [[tensorflow](https://github.com/xyf513/SpiderCNN)] [__`cls.`__ __`seg.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf)] Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search. [__`reg.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kejie_Li_Efficient_Dense_Point_ECCV_2018_paper.pdf)] Efficient Dense Point Cloud Object Reconstruction using Deformation Vector Fields. [__`rec.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dario_Rethage_Fully-Convolutional_Point_Networks_ECCV_2018_paper.pdf)] Fully-Convolutional Point Networks for Large-Scale Point Clouds. [[tensorflow](https://github.com/drethage/fully-convolutional-point-network)] [__`seg.`__ __`oth.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)] Deep Continuous Fusion for Multi-Sensor 3D Object Detection. [__`det.`__]
- [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)] HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration. [__`reg.`__]
- 
- [[AAAI]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16530/16302)] Learning Efficient Point Cloud Generation for Dense 3D Object Reconstruction. [[tensorflow](https://github.com/chenhsuanlin/3D-point-cloud-generation)] [__`rec.`__] :fire:
- 
- [[NeurIPS](https://papers.nips.cc/paper/7545-unsupervised-learning-of-shape-and-pose-with-differentiable-point-clouds)] Unsupervised Learning of Shape and Pose with Differentiable Point Clouds. [[tensorflow](https://github.com/eldar/differentiable-point-clouds)] [__`pos.`__]
- [[NeurIPS](https://papers.nips.cc/paper/7362-pointcnn-convolution-on-x-transformed-points)] PointCNN: Convolution On X-Transformed Points. [[tensorflow](https://github.com/yangyanli/PointCNN)][[pytorch](https://github.com/hxdengBerkeley/PointCNN.Pytorch)] [__`cls.`__ __`seg.`__] :fire:
- 
- [[SIGGRAPH](https://arxiv.org/abs/1803.10091)] Point Convolutional Neural Networks by Extension Operators. [[tensorflow](https://github.com/matanatz/pcnn)] [__`cls.`__ __`seg.`__]
- [[SIGGRAPH](https://arxiv.org/abs/1803.09263)] P2P-NET: Bidirectional Point Displacement Net for Shape Transform. [[tensorflow](https://github.com/kangxue/P2P-NET)] [__`oth.`__]
- [[SIGGRAPH Asia](https://arxiv.org/abs/1806.01759)] Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds. [[tensorflow](https://github.com/viscom-ulm/MCCNN)] [__`cls.`__ __`seg.`__ __`oth.`__]
- [[SIGGRAPH](https://arxiv.org/abs/1706.04496)] Learning local shape descriptors from part correspondences with multi-view convolutional networks. [[project](https://people.cs.umass.edu/~hbhuang/local_mvcnn/index.html)] [__`seg.`__ __`oth.`__] 
-
- [[MM](https://arxiv.org/abs/1808.07659)] PVNet: A Joint Convolutional Network of Point Cloud and Multi-View for 3D Shape Recognition. [__`cls.`__ __`rel.`__]
- [[MM](https://arxiv.org/abs/1806.02952)] RGCNN: Regularized Graph CNN for Point Cloud Segmentation. [[tensorflow](https://github.com/tegusi/RGCNN)] [__`seg.`__]
- [[MM](https://arxiv.org/abs/1804.10783)] Hybrid Point Cloud Attribute Compression Using Slice-based Layered Structure and Block-based Intra Prediction. [__`oth.`__]
-
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462884)] End-to-end Learning of Multi-sensor 3D Tracking by Detection. [__`det.`__ __`tra.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460837)] Multi-View 3D Entangled Forest for Semantic Segmentation and Mapping. [__`seg.`__ __`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462926)] SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud. [[tensorflow](https://github.com/priyankanagaraj1494/Squeezseg)] [__`seg.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461257)] Robust Real-Time 3D Person Detection for Indoor and Outdoor Applications. [__`det.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461048)] High-Precision Depth Estimation with the 3D LiDAR and Stereo Fusion. [__`dep.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461095)] Sampled-Point Network for Classification of Deformed Building Element Point Clouds. [__`cls.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460532)] Gemsketch: Interactive Image-Guided Geometry Extraction from Point Clouds. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460605)] Signature of Topologically Persistent Points for 3D Point Cloud Description. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461232)] A General Pipeline for 3D Detection of Vehicles. [__`det.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460716)] Robust and Fast 3D Scan Alignment Using Mutual Information. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460940)] Delight: An Efficient Descriptor for Global Localisation Using LiDAR Intensities. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460862)] Surface-Based Exploration for Autonomous 3D Modeling. [__`oth.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460554)] Deep Lidar CNN to Understand the Dynamics of Moving Vehicles. [__`oth.`__ __`aut.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460887)] Dex-Net 3.0: Computing Robust Vacuum Suction Grasp Targets in Point Clouds Using a New Analytic Model and Deep Learning. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460639)] Real-Time Object Tracking in Sparse Point Clouds Based on 3D Interpolation. [__`tra.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460825)] Robust Generalized Point Cloud Registration Using Hybrid Mixture Model. [__`reg.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461049)] A General Framework for Flexible Multi-Cue Photometric Point Cloud Registration. [__`reg.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461000)] Efficient Continuous-Time SLAM for 3D Lidar-Based Online Mapping. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461102)] Direct Visual SLAM Using Sparse Depth for Camera-LiDAR System. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460910)] Spatiotemporal Learning of Dynamic Gestures from 3D Point Cloud Data. [__`cls.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460204)] Asynchronous Multi-Sensor Fusion for 3D Mapping and Localization. [__`oth.`__]
- [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460834)] Complex Urban LiDAR Data Set. [[video](https://www.youtube.com/watch?v=IguZjmLf5V0&feature=youtu.be)] [__`dat.`__ __`oth.`__]
- 
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593693)] CalibNet: Geometrically Supervised Extrinsic Calibration using 3D Spatial Transformer Networks.[[tensorflow](https://github.com/epiception/CalibNet)] [__`oth.`__ __`aut.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593839)] Dynamic Scaling Factors of Covariances for Accurate 3D Normal Distributions Transform Registration. [__`reg.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593733)] A 3D Laparoscopic Imaging System Based on Stereo-Photogrammetry with Random Patterns. [__`rec.`__ __`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593558)] Robust Generalized Point Cloud Registration with Expectation Maximization Considering Anisotropic Positional Uncertainties. [__`reg.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594024)] Octree map based on sparse point cloud and heuristic probability distribution for labeled images. [__`oth.`__ __`aut.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593854)] PoseMap: Lifelong, Multi-Environment 3D LiDAR Localization. [__`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593953)] Scan Context: Egocentric Spatial Descriptor for Place Recognition Within 3D Point Cloud Map. [__`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594299)] LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain.[[code](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM)] [__`pos.`__ __`oth.`__] :fire:
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593741)] Classification of Hanging Garments Using Learned Features Extracted from 3D Point Clouds. [__`cls.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594362)] Stereo Camera Localization in 3D LiDAR Maps. [__`pos.`__ __`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594362)] Joint 3D Proposal Generation and Object Detection from View Aggregation. [__`det.`__] :star:
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594318)] Joint Point Cloud and Image Based Localization for Efficient Inspection in Mixed Reality. [__`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593910)] Edge and Corner Detection for Unorganized 3D Point Clouds with Application to Robotic Welding. [__`det.`__ __`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594175)] NDVI Point Cloud Generator Tool Using Low-Cost RGB-D Sensor. [[code](https://github.com/CTTCGeoLab/VI_ROS)][__`oth.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593837)] A 3D Convolutional Neural Network Towards Real-Time Amodal 3D Object Detection. [__`det.`__ __`pos.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594356)] Extracting Phenotypic Characteristics of Corn Crops Through the Use of Reconstructed 3D Models. [__`seg.`__ __`rec.`__]
- [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594514)] PCAOT: A Manhattan Point Cloud Registration Method Towards Large Rotation and Small Overlap. [__`reg.`__]
- 
- [[SENSORS](https://www.mdpi.com/1424-8220/18/10/3337)] SECOND: Sparsely Embedded Convolutional Detection. [[pytorch](https://github.com/traveller59/second.pytorch)] [__`det.`__ __`aut.`__] :fire:
-
- [[ACCV](https://arxiv.org/abs/1803.07289)] Flex-Convolution (Million-Scale Point-Cloud Learning Beyond Grid-Worlds). [[tensorflow](https://github.com/cgtuebingen/Flex-Convolution)] [__`seg.`__]
-
- [[3DV](https://arxiv.org/abs/1808.00671)] PCN: Point Completion Network. [[tensorflow](https://github.com/TonythePlaneswalker/pcn)] [__`reg.`__ __`oth.`__ __`aut.`__]
-
- [[ICASSP](https://arxiv.org/abs/1812.01711)] A Graph-CNN for 3D Point Cloud Classification. [[tensorflow](https://github.com/maggie0106/Graph-CNN-in-3D-Point-Cloud-Classification)] [__`cls.`__] :fire:
-
- [[arXiv](https://arxiv.org/abs/1807.00652)] PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation. [[tensorflow](https://github.com/MVIG-SJTU/pointSIFT)] [__`seg.`__] :fire:
- [[arXiv](https://arxiv.org/abs/1801.07829)] Dynamic Graph CNN for Learning on Point Clouds. [[tensorflow](https://github.com/WangYueFt/dgcnn)][[pytorch](https://github.com/muhanzhang/pytorch_DGCNN)] [__`cls.`__ __`seg.`__] :fire: :star:
- [[arXiv](https://arxiv.org/abs/1805.07872)] Spherical Convolutional Neural Network for 3D Point Clouds. [__`cls.`__]
- [[arXiv](https://arxiv.org/abs/1811.07605)] Adversarial Autoencoders for Generating 3D Point Clouds. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1811.11209)] Iterative Transformer Network for 3D Point Cloud. [__`cls.`__ __`seg.`__ __`pos.`__]
- [[arXiv](https://arxiv.org/abs/1811.12543)] Topology-Aware Surface Reconstruction for Point Clouds. [__`rec.`__]
- [[arXiv](https://arxiv.org/abs/1812.01402)] Inferring Point Clouds from Single Monocular Images by Depth Intermediation. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1812.04302)] Deep RBFNet: Point Cloud Feature Learning using Radial Basis Functions. [__`cls.`__]
- [[arXiv](https://arxiv.org/abs/1812.05276)] IPOD: Intensive Point-based Object Detector for Point Cloud. [__`det.`__]
- [[arXiv](https://arxiv.org/abs/1812.07050)] 3D Point Cloud Learning for Large-scale Environment Analysis and Place Recognition. [__`rel.`__ __`oth.`__]
- [[arXiv](https://arxiv.org/abs/1812.11017)] Deflecting 3D Adversarial Point Clouds Through Outlier-Guided Removal. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1812.11383)] Feature Preserving and Uniformity-controllable Point Cloud Simplification on Graph. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1901.01060)] POINTCLEANNET: Learning to Denoise and Remove Outliers from Dense Point Clouds. [[pytorch](https://github.com/mrakotosaon/pointcleannet)] [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1803.06199)] Complex-YOLO: Real-time 3D Object Detection on Point Clouds. [[pytorch](https://github.com/AI-liu/Complex-YOLO)] [__`det.`__ __`aut.`__] :fire:
- [[arxiv](https://arxiv.org/abs/1811.03818)] RoarNet: A Robust 3D Object Detection based on RegiOn Approximation Refinement. [[tensorflow](https://github.com/Kiwoo/RoarNet)] [__`det.`__ __`aut.`__]

---
## 2019
- [[CVPR](http://export.arxiv.org/abs/1904.07601)] Relation-Shape Convolutional Neural Network for Point Cloud Analysis. [[pytorch](https://github.com/Yochengliu/Relation-Shape-CNN)] [__`cls.`__ __`seg.`__ __`oth.`__]
- [[CVPR](https://raoyongming.github.io/files/SFCNN.pdf)] Spherical Fractal Convolutional Neural Networks for Point Cloud Recognition. [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1811.11397)] DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds. [[code](https://ai4ce.github.io/DeepMapping/)] [__`reg.`__]
- [[CVPR](https://arxiv.org/abs/1812.07179)] Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving. [[code](https://github.com/mileyan/pseudo_lidar)] [__`det.`__ __`dep.`__ __`aut.`__]
- [[CVPR](https://arxiv.org/abs/1812.04244)] PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud. [[pytorch](https://github.com/sshaoshuai/PointRCNN)] [__`det.`__ __`aut.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1809.07016)] Generating 3D Adversarial Point Clouds. [[code](https://github.com/xiangchong1/3d-adv-pc)] [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1904.03375v1)] Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling. [__`cls.`__ __`seg.`__]
- [[CVPR](http://export.arxiv.org/abs/1904.08017)] A-CNN: Annularly Convolutional Neural Networks on Point Clouds. [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1811.07246)] PointConv: Deep Convolutional Networks on 3D Point Clouds. [[tensorflow](https://github.com/DylanWusee/pointconv)] [__`cls.`__ __`seg.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1812.11647)] Path-Invariant Map Networks. [[tensorflow](https://github.com/zaiweizhang/path_invariance_map_network)] [__`seg.`__ __`oth.`__]
- [[CVPR](https://arxiv.org/abs/1812.02713)] PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding. [[code](https://github.com/daerduoCarey/partnet_dataset)] [__`dat.`__ __`seg.`__]
- [[CVPR](http://export.arxiv.org/abs/1901.00680)] GeoNet: Deep Geodesic Networks for Point Cloud Analysis. [__`cls.`__ __`rec.`__ __`oth.`__]
- [[CVPR](https://arxiv.org/abs/1902.09852)] Associatively Segmenting Instances and Semantics in Point Clouds. [[tensorflow](https://github.com/WXinlong/ASIS)] [__`seg.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1811.08988)] Supervised Fitting of Geometric Primitives to 3D Point Clouds. [[tensorflow](https://github.com/csimstu2/SPFN)] [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1903.00343)] Octree guided CNN with Spherical Kernels for 3D Point Clouds. [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. [[pytorch](https://github.com/hmgoforth/PointNetLK)] [__`reg.`__]
- [[CVPR](https://arxiv.org/abs/1904.00699v1)] JSIS3D: Joint Semantic-Instance Segmentation of 3D Point Clouds with Multi-Task Pointwise Networks and Multi-Value Conditional Random Fields. [[pytorch](https://github.com/pqhieu/JSIS3D)] [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/1904.02113)] Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning. [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/1812.05784)] PointPillars: Fast Encoders for Object Detection from Point Clouds. [[pytorch](https://github.com/nutonomy/second.pytorch)] [__`det.`__] :fire:
- [[CVPR](https://arxiv.org/abs/1811.11286)] Patch-based Progressive 3D Point Set Upsampling. [[tensorflow](https://github.com/yifita/3PU)] [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1904.09793)] PCAN: 3D Attention Map Learning Using Contextual Information for Point Cloud Based Retrieval. [[code](https://github.com/XLechter/PCAN)] [__`rel.`__]
- [[CVPR](https://arxiv.org/abs/1903.00709)] PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation. [[pytorch](https://github.com/FoggYu/PartNet)] [__`dat.`__ __`seg.`__] 
- [[CVPR](https://arxiv.org/abs/1806.02170)] PointFlowNet: Learning Representations for Rigid Motion Estimation from Point Clouds. [[code](https://github.com/aseembehl/pointflownet)] [__`det.`__ __`dat.`__ __`oth.`__] 
- [[CVPR](https://arxiv.org/abs/1904.03483)] SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. [[matlab](https://github.com/intellhave/SDRSAC)] [__`reg.`__]
- [[CVPR](https://arxiv.org/abs/1903.04019)] Deep Reinforcement Learning of Volume-guided Progressive View Inpainting for 3D Point Scene Completion from a Single Depth Image. [__`rec.`__ __`oth.`__]
- [[CVPR](https://arxiv.org/abs/1904.03461)] Embodied Question Answering in Photorealistic Environments with Point Cloud Perception. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1812.10775v1)] 3D Point-Capsule Networks. [[pytorch](https://github.com/yongheng1991/3D-point-capsule-networks)] [__`cls.`__ __`rec.`__ __`oth.`__]
- [[CVPR](http://export.arxiv.org/abs/1904.08755)] 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/1811.06879v2)] The Perfect Match: 3D Point Cloud Matching with Smoothed Densities. [[tensorflow](https://github.com/zgojcic/3DSmoothNet)] [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1811.10136)] FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. [[code](https://bitbucket.org/gaowei19951004/poser/src/master/)] [__`reg.`__]
- [[CVPR](https://arxiv.org/abs/1806.01411)] FlowNet3D: Learning Scene Flow in 3D Point Clouds. [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1811.07782)] Modeling Local Geometric Structure of 3D Point Clouds using Geo-CNN. [__`cls.`__ __`det.`__]
- [[CVPR](http://www.linliang.net/wp-content/uploads/2019/04/CVPR2019_PointClound.pdf)] ClusterNet: Deep Hierarchical Cluster Network with Rigorously Rotation-Invariant Representation for Point Cloud Analysis. [__`cls.`__]
- [[CVPR](http://jiaya.me/papers/pointweb_cvpr19.pdf)] PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing. [__`cls.`__ __`seg.`__]
- [[CVPR](https://arxiv.org/abs/1904.12304)] RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion. [[code](https://github.com/iSarmad/RL-GAN-Net)] [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Robust & Efficient Point Cloud Registration using PointNet. [[pytorch](https://github.com/hmgoforth/PointNetLK)] [__`reg.`__]
- [[CVPR](https://www.researchgate.net/publication/332240602_Robust_Point_Cloud_Based_Reconstruction_of_Large-Scale_Outdoor_Scenes)] Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes. [[code](https://github.com/ziquan111/RobustPCLReconstruction)] [__`rec.`__]
- [[CVPR](https://arxiv.org/abs/1812.00709)] Nesti-Net: Normal Estimation for Unstructured 3D Point Clouds using Convolutional Neural Networks. [[tensorflow](https://github.com/sitzikbs/Nesti-Net)] [__`oth.`__]
- [[CVPR](https://arxiv.org/abs/1812.03320)] GSPN: Generative Shape Proposal Network for 3D Instance Segmentation in Point Cloud. [__`seg.`__]
- [[CVPR](https://engineering.purdue.edu/~jshan/publications/2018/Lei%20Wang%20Graph%20Attention%20Convolution%20for%20Point%20Cloud%20Segmentation%20CVPR2019.pdf)] Graph Attention Convolution for Point Cloud Segmentation. [__`seg.`__]
- [[CVPR](https://arxiv.org/abs/1812.02050)] Point-to-Pose Voting based Hand Pose Estimation using Residual Permutation Equivariant Layer. [__`pos.`__]
- [[CVPR](https://arxiv.org/abs/1903.08701v1)] LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving. [__`det.`__ __`aut.`__]
- 
- [[ICLR](https://openreview.net/forum?id=SJeXSo09FQ)] Learning Localized Generative Models for 3D Point Clouds via Graph Convolution. [__`oth.`__]
-
- [[AAAI](https://arxiv.org/abs/1811.11731)] CAPNet: Continuous Approximation Projection For 3D Point Cloud Reconstruction Using 2D Supervision. [[code](https://github.com/val-iisc/capnet)] [__`rec.`__] 
- [[AAAI](https://arxiv.org/abs/1811.02565)] Point2Sequence: Learning the Shape Representation of 3D Point Clouds with an Attention-based Sequence to Sequence Network. [[tensorflow](https://github.com/liuxinhai/Point2Sequence)] [__`cls.`__ __`seg.`__]
- [[AAAI](https://par.nsf.gov/biblio/10086163)] Point Cloud Processing via Recurrent Set Encoding. [__`cls.`__]
- [[AAAI](https://arxiv.org/abs/1812.00333)] PVRNet: Point-View Relation Neural Network for 3D Shape Recognition. [[pytorch](https://github.com/Hxyou/PVRNet)] [__`cls.`__ __`rel.`__]
-
- [[ICRA](https://arxiv.org/abs/1904.00319)] Discrete Rotation Equivariance for Point Cloud Recognition. [[pytorch](https://github.com/lijx10/rot-equ-net)] [__`cls.`__]
- [[ICRA](https://arxiv.org/abs/1809.08495)] SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud. [[tensorflow](https://github.com/xuanyuzhou98/SqueezeSegV2)] [__`seg.`__ __`aut.`__]
-
- [[arXiv](https://arxiv.org/abs/1901.02532)] Fast 3D Line Segment Detection From Unorganized Point Cloud. [__`det.`__]
- [[arXiv](https://arxiv.org/abs/1812.01687)] Point-Cloud Saliency Maps. [[tensorflow](https://github.com/tianzheng4/PointCloud-Saliency-Maps)] [__`cls.`__ __`oth.`__]
- [[arXiv](https://export.arxiv.org/abs/1901.03006)] Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers. [[code](https://github.com/Daniel-Liu-c0deb0t/3D-Neural-Network-Adversarial-Attacks)] [__`oth.`__]
- [[arxiv](https://arxiv.org/abs/1901.08396)] Context Prediction for Unsupervised Deep Learning on Point Clouds. [__`cls.`__ __`seg.`__]
- [[arXiv](http://export.arxiv.org/abs/1901.09280)] Points2Pix: 3D Point-Cloud to Image Translation using conditional Generative Adversarial Networks. [__`oth.`__]
- [[arXiv](http://export.arxiv.org/abs/1901.09394)] NeuralSampler: Euclidean Point Cloud Auto-Encoder and Sampler. [__`cls.`__ __`oth.`__]
- [[arXiv](https://arxiv.org/abs/1902.05247)] 3D Graph Embedding Learning with a Structure-aware Loss Function for Point Cloud Semantic Instance Segmentation. [__`seg.`__]
- [[arXiv](https://arxiv.org/abs/1902.10272)] Zero-shot Learning of 3D Point Cloud Objects. [[code](https://github.com/alichr/Zero-shot-Learning-of-3D-Point-Cloud-Objects)] [__`cls.`__]
- [[arXiv](https://arxiv.org/abs/1903.09847)] Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud. [__`det.`__ __`aut.`__]
- [[arXiv](https://arxiv.org/abs/1903.01695)] Real-time Multiple People Hand Localization in 4D Point Clouds. [__`det.`__ __`oth.`__]
- [[arXiv](https://arxiv.org/abs/1903.02858)] Variational Graph Methods for Efficient Point Cloud Sparsification. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1903.05807)] Neural Style Transfer for Point Clouds. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1903.07918)] OREOS: Oriented Recognition of 3D Point Clouds in Outdoor Scenarios. [__`pos.`__ __`oth.`__]
- [[arXiv](https://arxiv.org/abs/1903.10750)] FVNet: 3D Front-View Proposal Generation for Real-Time Object Detection from Point Clouds. [[code](https://github.com/LordLiang/FVNet)] [__`det.`__ __`aut.`__]
- [[arXiv](https://arxiv.org/abs/1904.00069)] Unpaired Point Cloud Completion on Real Scans using Adversarial Training. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1904.00229)] USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds. [[code](https://github.com/lijx10/USIP)] [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1904.00230)] MortonNet: Self-Supervised Learning of Local Features in 3D Point Clouds. [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1904.00817)] DeepPoint3D: Learning Discriminative Local Descriptors using Deep Metric Learning on 3D Point Clouds. [__`cls.`__ __`rel.`__ __`oth.`__]
- [[arXiv](https://arxiv.org/abs/1904.01416)] A Dataset for Semantic Segmentation of Point Cloud Sequences. [__`dat.`__ __`seg.`__]
- [[arXiv](http://arxiv.org/abs/1904.02375)] ConvPoint: Generalizing discrete convolutions for unstructured point clouds. [[pytorch](https://github.com/aboulch/ConvPoint)] [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1904.04427)] 3D Point Cloud Denoising via Deep Neural Network based Local Surface Estimation. [[code](https://github.com/chaojingduan/Neural-Projection)] [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1904.07537)] Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds. [[pytorch](https://github.com/AI-liu/Complex-YOLO)] [__`det.`__ __`tra.`__ __`aut.`__] :fire:
- [[arXiv](https://arxiv.org/abs/1904.10795)] Graph-based Inpainting for 3D Dynamic Point Clouds. [__`oth.`__]
- [[arXiv](https://arxiv.org/abs/1903.11027)] nuScenes: A multimodal dataset for autonomous driving. [[link](https://www.nuscenes.org/overview)] [__`dat.`__ __`det.`__ __`tra.`__ __`aut.`__]
- [[arXiv](https://arxiv.org/abs/1901.08373)] 3D Backbone Network for 3D Object Detection. [[code](https://github.com/Benzlxs/tDBN)] [__`det.`__ __`aut.`__]
- [[arXiv](https://arxiv.org/abs/1904.08889)] KPConv: Flexible and Deformable Convolution for Point Clouds. [[tensorflow](https://github.com/HuguesTHOMAS/KPConv)] [__`cls.`__ __`seg.`__]
- [[arXiv](https://arxiv.org/abs/1904.09664v1)] Deep Hough Voting for 3D Object Detection in Point Clouds. [__`det.`__]
- [[arXiv](https://arxiv.org/abs/1811.07605v3)] Adversarial Autoencoders for Compact Representations of 3D Point Clouds. [[pytorch](https://github.com/MaciejZamorski/3d-AAE)] [__`rel.`__ __`oth.`__]

<h1> 

```diff
- Datasets
```

</h1>

- [[KITTI](http://www.cvlibs.net/datasets/kitti/)] The KITTI Vision Benchmark Suite. [__`det.`__]
- [[ModelNet](http://modelnet.cs.princeton.edu/)] The Princeton ModelNet . [__`cls.`__]
- [[ShapeNet](https://www.shapenet.org/)]  A collaborative dataset between researchers at Princeton, Stanford and TTIC. [__`seg.`__]
- [[PartNet](https://shapenet.org/download/parts)] The PartNet dataset provides fine grained part annotation of objects in ShapeNetCore. [__`seg.`__]
- [[PartNet](http://kevinkaixu.net/projects/partnet.html)] PartNet benchmark from Nanjing University and National University of Defense Technology. [__`seg.`__]
- [[S3DIS](http://buildingparser.stanford.edu/dataset.html#Download)] The Stanford Large-Scale 3D Indoor Spaces Dataset. [__`seg.`__]
- [[ScanNet](http://www.scan-net.org/)] Richly-annotated 3D Reconstructions of Indoor Scenes. [__`cls.`__ __`seg.`__]
- [[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. [__`reg.`__]
- [[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . [__`cls.`__ __`seg.`__ __`reg.`__]
- [[Princeton Shape Benchmark](http://shape.cs.princeton.edu/benchmark/)] The Princeton Shape Benchmark.
- [[SYDNEY URBAN OBJECTS DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml)] This dataset contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles, pedestrians, signs and trees. [__`cls.`__ __`match.`__]
- [[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community with the aim to facilitate result evaluations and comparisons. [__`cls.`__ __`match.`__ __`reg.`__ __`det`__]
- [[Large-Scale Point Cloud Classification Benchmark(ETH)](http://www.semantic3d.net/)] This benchmark closes the gap and provides a large labelled 3D point cloud data set of natural scenes with over 4 billion points in total. [__`cls.`__]
- [[Robotic 3D Scan Repository](http://asrl.utias.utoronto.ca/datasets/3dmap/)] The Canadian Planetary Emulation Terrain 3D Mapping Dataset is a collection of three-dimensional laser scans gathered at two unique planetary analogue rover test facilities in Canada.  
- [[Radish](http://radish.sourceforge.net/)] The Robotics Data Set Repository (Radish for short) provides a collection of standard robotics data sets.
- [[IQmulus & TerraMobilita Contest](http://data.ign.fr/benchmarks/UrbanAnalysis/#)] The database contains 3D MLS data from a dense urban environment in Paris (France), composed of 300 million points. The acquisition was made in January 2013. [__`cls.`__ __`seg.`__ __`det.`__]
- [[Oakland 3-D Point Cloud Dataset](http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/)] This repository contains labeled 3-D point cloud laser data collected from a moving platform in a urban environment.
- [[Robotic 3D Scan Repository](http://kos.informatik.uni-osnabrueck.de/3Dscans/)] This repository provides 3D point clouds from robotic experiments，log files of robot runs and standard 3D data sets for the robotics community.
- [[Ford Campus Vision and Lidar Data Set](http://robots.engin.umich.edu/SoftwareData/Ford)] The dataset is collected by an autonomous ground vehicle testbed, based upon a modified Ford F-250 pickup truck. 
- [[The Stanford Track Collection](https://cs.stanford.edu/people/teichman/stc/)] This dataset contains about 14,000 labeled tracks of objects as observed in natural street scenes by a Velodyne HDL-64E S2 LIDAR.
- [[PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html)] Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild. [__`pos.`__ __`det.`__]
- [[3D MNIST](https://www.kaggle.com/daavoo/3d-mnist)] The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition. [__`cls.`__]
- [[WAD](http://wad.ai/)] This dataset is provided by Baidu Inc.
- [[nuScenes](https://d3u7q4379vrm7e.cloudfront.net/object-detection)] The nuScenes dataset is a large-scale autonomous driving dataset.
- [[PreSIL](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/precise-synthetic-image-and-lidar-presil-dataset-autonomous)] Precise Synthetic Image and LiDAR (PreSIL) Dataset for Autonomous Vehicle Perception. It consists of over 50,000 instances and includes high-definition images with full resolution depth information, semantic segmentation (images), point-wise segmentation (point clouds), ground point labels (point clouds), and detailed annotations for all vehicles and people. [[paper](https://arxiv.org/abs/1905.00160)] [__`det.`__ __`aut.`__]
- [[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. [__`reg.`__ __`rec.`__ __`oth.`__]
