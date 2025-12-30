## GT Data structure

``` python
- b2d_map_infos.pkl # 高精地图文件
- v1/
    - scenario_name/Town[id]_route[id]_weather[id]/
        - anno
            - b2d_infos.pkl # gt注释文件
        - camera
            - rgb_back
                - 00000.jpg # rgb图片文件
            - rgb_back_left
                - 00000.jpg
            - rgb_back_right
                - 00000.jpg
            - rgb_front
                - 00000.jpg
            - rgb_front_left
                - 00000.jpg
            - rgb_front_right
                - 00000.jpg
            - rgb_top_down
                - 00000.jpg
```

## GT Anno Structure
**TODO: GT注释文件b2d_infos.pkl是一个列表，每一项代表了一帧的注释。
注释里涉及的坐标基本都是已经转到了lidar坐标系下，例如边界框，轨迹等，在可视化时直接画就行。
后续可能会新增字段。**
``` shell
    - scene_token
    - sample_token # 场景中当前帧的标识
    - frame_idx # 当前帧idx
    - ego2global_translation # 自车到global的位移，也是自车在世界坐标系中的位置
    - ego2global_rotation
    - timestamp
    - map_location # 这一帧所处的地图位置，为了后续去高精地图中检索对应的地图点
    - can_bus # [3+4+3+3+3+2]: ego2global_translation, ego2global_rotation(四元数形式), 自车线性速度, 自车加速度, 自车角速度, yaw, 弧度制的yaw
    - ego_his_trajs # [2, 2]: 自车历史2帧的轨迹
    - ego_fut_trajs # [6, 2]: 未来6帧的轨迹，是offset形式，获取实际的轨迹需要cumsum一下
    - ego_fut_masks
    - ego_lcf_feat # 自车训练过程中所需的一些属性，可视化可能用不到。[2+2+1+1+1+1+1]: 正北正东两个方向的速度分量，xy轴(IMU坐标系)加速度分量，yaw角的角速度，l，w，线性速度，轮速
    - fut_valid_flag # bool: 这一帧未来有无轨迹，如果为False类似于静止
    - ego_fut_cmd # [6]: 当前帧的高级控制指令的onehot编码，不同数据集可能维度不同
    - ego_fut_cmd_name # 高级指令的名字
    - num_points # [b]: b是gt_bboxes_3d的个数，存的是bboxes的点云数目，用来生成mask
    - sensors
        - CAM_XXXX
            - intrinsic # The intrinsic of camera.
            - cam2ego # Transformation from camera coordinates to ego_vehicle coordinates
            - data_path # 对应的传感器数据路径，这里是对于数据集目录的相对路径
            - lidar2cam # lidar到此camera的变换
            - lidar2img # lidar到此camera像素坐标系的变换
        - LIDAR_TOP
            - map_pts_world2lidar # world坐标系到lidar坐标系的变换，注意这里这个变换是专门用来将全局坐标下的地图点转到lidar坐标系的，其它需要从global->lidar请使用ego2global和lidar2ego获取。
            - lidar2ego # lidar到自车坐标系的变换
    - gt_boxes_anno_info
        - gt_bboxes_3d # [b, 10]：x, y, z, w, l, h, yaw, vx, vy, gt_id. 其中yaw是SECOND格式的，真实的与lidar的x轴夹角要取反再减90度
        - gt_names
        - gt_fut_trajs # [b, 6, 2]，未来6帧(3s)的轨迹，是offset形式，获取实际的轨迹需要cumsum一下
        - gt_fut_masks # [b, 6]
        - attr_labels # [b, 12+6+1+9+6]: fut_trajs, fut_mask, fut_goal, agent_lcf_feat, future_yaw_offset
        - interested_id # 这一帧感兴趣目标的全局id
```
## HD-Map Data Structure
``` shell
  # b2d_map_infos.pkl包含了所有地图的信息。最外层是一个dict，键值是每个地图的名字
  - town_id  # CARLA town id
    # 每一个town_id同样是一个字典:
    - lane_points # 是一个列表，里边每一项是某个道路的点云集合
        -0000 # [pts_num, 3]存储了该道路所有点云的点坐标
    - lane_sample_points # 同样是一个列表，每一项和lane_points对应的是同一条道路，但存储的是采样后的点云点坐标。
    - lane_types  # 同样是一个列表，每一项和lane_points对应的是同一条道路，存储的是道路的类型
    - trigger_volumes_points # 这三个字段存储的内容和前三个格式上完全一致，只是这里存的不是道路的点云，而是一些交通信号标志区域的点云，一般是个多边形区域
    - trigger_volumes_sample_points # 多边形区域的中心点
    - traigger_volumes_types
    '再次提醒高精地图下的点都是carla原始全局坐标系下的点，需要使用anno[senors][LIDAR_TOP][map_pts_world2lidar]转换到lidar坐标系下'
    
```
## coordinates
其实给的数据中除了高精地图是全局坐标系下的点需要转换，其它的都已经是局部坐标系下的坐标点了。但这里还是标注下后续打算
固定的标准坐标系形式：

世界坐标系：东北天

自车：前左上

lidar：右前上

camera： 右下前

IMU： 前左上


## Pred Data structure

``` python
- results/
    - scenario_name/Town[id]_route[id]_weather[id]/
        - metrics.pkl # 评价指标文件
        - results.pkl # 预测结果文件
```

## Anno Structure
**results.pkl**
``` shell
    - scene_token
    - sample_token # 场景中当前帧的标识
    - frame_idx # 当前帧idx
    - ego_fut_trajs # [6, 6, 2]: 未来6帧的多模态轨迹，是offset形式，获取实际的轨迹需要cumsum一下
    - ego_fut_cmd # [6]: 当前帧的高级控制指令的onehot编码，不同数据集可能维度不同
    - pred_boxes_anno_info
        - gt_bboxes_3d # [b, 9]：x, y, z, w, l, h, yaw, vx, vy. 其中yaw是SECOND格式的，真实的与lidar的x轴夹角要取反再减90度
        - gt_names
        - gt_fut_trajs # [b, 6, 6, 2]，未来6帧(3s)的多模态轨迹，是offset形式，获取实际的轨迹需要cumsum一下
    - text # QA相关，列表中每一项包含了一条QA（仅VLA模型有这一条）
```

**metrics.pkl**
``` shell
    - frame_metrics # 每一帧单独的评价指标列表
        - sample_token
        - metrics
            - planning_L2_1s # 当前帧planning的指标
            - planning_L2_2s
            - planning_L2_3s
            - perception_xxx # 当前帧的perception指标，目前没有相关字段
            - motion_xxx # 当前帧的motion指标，目前没有相关字段
        - metrics_valid_flag
            - planning_L2_1s # bool 表示这一帧的对应指标是否有效，如果为False意味着这一帧的此指标不会被纳入整个场景最终的指标计算过程
            - planning_L2_2s
            - planning_L2_3s
            - perception_xxx # 目前没有相关字段
            - motion_xxx # 目前没有相关字段
            
    - perception # 这个场景的平均感知指标，当前为None
    - motion # 当前为None
    - planning # 这个场景的平均规划指标
        - mean_planning_L2_1s
        - mean_planning_L2_2s
        - mean_planning_L2_3s
```