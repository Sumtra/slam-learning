import cv2
import numpy as np

class FREAKFeatureDetector:
    def __init__(self):
        # 使用ORB替代FREAK和SIFT
        self.orb = cv2.ORB_create()

    def detect_and_match(self, img1_path, img2_path):
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("无法读取图像文件")
            
        # 使用ORB检测特征点和计算描述符
        keypoints1, descriptors1 = self.orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(img2, None)
        
        # 创建汉明距离匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # 进行特征匹配
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # 应用比率测试筛选好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # 输出特征点和匹配信息
        print(f"图像1特征点数量: {len(keypoints1)}")
        print(f"图像2特征点数量: {len(keypoints2)}")
        print(f"匹配点数量: {len(good_matches)}")
        
        # 绘制匹配结果
        result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 保存结果到文件
        output_path = 'feature_matching_result.jpg'
        cv2.imwrite(output_path, result)
        print(f"匹配结果已保存至: {output_path}")

def main():
    detector = FREAKFeatureDetector()
    # 设置图像路径
    img1_path = r"E:\slam image\1.jpg"
    img2_path = r"E:\slam image\2.jpg"
    
    # 执行特征检测和匹配
    detector.detect_and_match(img1_path, img2_path)

if __name__ == "__main__":
    main()
