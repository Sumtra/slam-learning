import cv2
import numpy as np
import time

class FeatureDetector:
    def __init__(self):
        try:
            self.sift = cv2.SIFT_create()
        except:
            # 如果SIFT不可用，可能需要安装opencv-contrib-python
            raise RuntimeError("无法创建SIFT检测器，请确保已安装opencv-contrib-python")

    def detect_sift(self, filename):
        img = cv2.imread(filename)
        if img is None:
            raise FileNotFoundError(f"无法读取图像文件: {filename}")
            
        # 转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测SIFT特征点和描述符
        keypoints, descriptors = self.sift.detectAndCompute(gray_img, None)
        return img, keypoints, descriptors

def main():
    detector = FeatureDetector()
    
    # 读取两张图片
    img1_path = r"E:\slam image\1.jpg"
    img2_path = r"E:\slam image\2.jpg"
    
    # SIFT特征检测
    img1, keypoints1, descriptors1 = detector.detect_sift(img1_path)
    img2, keypoints2, descriptors2 = detector.detect_sift(img2_path)
    
    # 在原图上绘制特征点
    img1_with_kp = cv2.drawKeypoints(img1, keypoints1, None, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv2.drawKeypoints(img2, keypoints2, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用Lowe比率测试筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # 输出特征点和匹配信息
    print(f"图像1特征点数量: {len(keypoints1)}")
    print(f"图像2特征点数量: {len(keypoints2)}")
    print(f"匹配点数量: {len(good_matches)}")
    
    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1_with_kp, keypoints1, 
                                 img2_with_kp, keypoints2, 
                                 good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 保存匹配结果
    output_path = 'sift_matches_result.jpg'
    cv2.imwrite(output_path, img_matches)
    print(f"匹配结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
