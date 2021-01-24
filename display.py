import cv2

video = cv2.VideoCapture(0)  # 打开摄像头

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频存储的格式
fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
# 视频的宽高
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), \
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('video.mp4', fourcc, fps, size)  # 视频存储

while out.isOpened():
    ret, img = video.read()  # 开始使用摄像头读数据，返回ret为true，img为读的图像
    if ret is False:  # ret为false则关闭
        exit()
    out.write(img)  # 将捕捉到的图像存储
    # 按esc键退出程序
    if cv2.waitKey(1) & 0xFF == 27:
        video.release()  # 关闭摄像头
        break
