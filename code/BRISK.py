import cv2
import numpy as np

class BRISKFeatureDetector:
    def __init__(self):
        # 创建BRISK检测器
        self.brisk = cv2.BRISK_create()
        
    def detect_and_match(self, img1_path, img2_path):
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("无法读取图像文件")
            
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点并计算描述子
        keypoints1, descriptors1 = self.brisk.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = self.brisk.detectAndCompute(gray2, None)
        
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 选取最佳匹配
        good_matches = matches[:50]  # 取前50个最佳匹配点
        
        # 输出特征点和匹配信息
        print(f"图像1特征点数量: {len(keypoints1)}")
        print(f"图像2特征点数量: {len(keypoints2)}")
        print(f"匹配点数量: {len(good_matches)}")
        
        # 绘制匹配结果
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 添加保存图片的代码:
        output_path = 'brisk_matches.jpg'
        cv2.imwrite(output_path, img_matches)
        print(f"匹配结果已保存至: {output_path}")

def main():
    detector = BRISKFeatureDetector()
    
    # 读取并处理图片
    img1_path = r"E:\slam image\1.jpg"
    img2_path = r"E:\slam image\2.jpg"
    
    detector.detect_and_match(img1_path, img2_path)

if __name__ == "__main__":
    main()
