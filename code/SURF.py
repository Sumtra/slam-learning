import cv2
import numpy as np

class SURFFeatureDetector:
    def __init__(self):
        try:
            # 创建SURF对象
            self.surf = cv2.xfeatures2d.SURF_create(400)  # 海塞矩阵阈值设为400
        except cv2.error:
            raise Exception("无法创建SURF检测器。请确保已安装 opencv-contrib-python")
        
        self.surf.setExtended(False)  # 不使用扩展描述符
        self.surf.setUpright(False)   # 不使用直立SURF
        self.surf.setNOctaves(4)      # 金字塔组数为4
        self.surf.setNOctaveLayers(3) # 每组层数为3

    def detect_and_match(self, img1_path, img2_path):
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("无法读取图像文件")
            
        # 将两张图片调整为相同大小
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点和计算描述符
        keypoints1, descriptors1 = self.surf.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = self.surf.detectAndCompute(gray2, None)
        
        # 创建FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 进行特征匹配
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # 应用Lowe比率测试筛选好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # 0.7是一个经验值
                good_matches.append(m)
        
        # 输出特征点和匹配信息
        print(f"图像1特征点数量: {len(keypoints1)}")
        print(f"图像2特征点数量: {len(keypoints2)}")
        print(f"匹配点数量: {len(good_matches)}")
        
        # 绘制匹配结果
        result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 保存结果到文件
        cv2.imwrite('surf_matches_result.jpg', result)
        print("匹配结果已保存为 'surf_matches_result.jpg'")

def main():
    detector = SURFFeatureDetector()
    # 设置图像路径
    img1_path = r"E:\slam image\1.jpg"
    img2_path = r"E:\slam image\2.jpg"
    detector.detect_and_match(img1_path, img2_path)

if __name__ == "__main__":
    main()
