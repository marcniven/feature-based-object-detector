import cv2
import numpy as np
from pathlib import Path

class ObjectDetector:

    def __init__(self, home):
        self.home = Path(home)
        self.objects = self.home / "Objects"  # Object Folder
        self.scenes = self.home / "Scenes"    # Scene Folder
        self.detected_objects = self.home / "Detected Objects" #Detected Objects Folder
        self.detected_objects.mkdir(exist_ok=True)
        # Dictionary for object names
        self.object_map = {
                "O1":   "Book",
                "O2":    "Game Controller",
                "O3":    "Mug",
                "O4":    "Phone",
                "O5":    "Ruler",
                "O6":    "Calculator",
                "O7":    "Waterbottle",
                "O8":    "Banana",
                "O9":    "Eraser",
                "O10":    "Camera"
        }

        self.object_features = {}
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    # Load and compute features for all object images.
    def load_objects(self):
        for i in range(1, 11):
            img = self.objects / f"O{i}.png"
            image = cv2.imread(str(img))
            if image is None:
                raise FileNotFoundError(f"Cannot load object image: {img}")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypts, descriptors = self.sift.detectAndCompute(gray, None)

            self.object_features[f"O{i}"] = {
                'keypoints': keypts,
                'descriptors': descriptors,
                'image': image,
                'shape': gray.shape
            }

            print("Object: ",i)

    # Detect objects in a scene, annotate them, and save the result.
    def detect_and_annotate(self, scene, threshold=10, pano=False):

        # Preprocess the image for feature detection
        scene_img = cv2.imread(str(scene))

        scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
        keypts, descriptors = self.sift.detectAndCompute(scene_gray, None)

        detected_objects = []
        for obj_name, obj_data in self.object_features.items():

            # If no features are found
            if descriptors is None or obj_data['descriptors'] is None:
                continue

            # Identify matches between features
            matches = self.flann.knnMatch(
                obj_data['descriptors'], descriptors, k=2
            )

            # Filter out matches using Lowe's Filter
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]


            # If there are more than 10 good matches then apply homopgraphy
            if len(good_matches) >= threshold:

                # passing in extra data such as scene_img, obj_name, and obj_data to allow for the Bounding Box
                is_valid, H = self.homography(obj_data['keypoints'], keypts, good_matches, scene_img, obj_name, obj_data)

                # If there are above 23 inliers and the matrix (used for bounding box) exists then added it to detected objects
                if is_valid and H is not None:
                    detected_objects.append(obj_name)

        output = self.detected_objects / f"{scene.stem}_bb.jpg"
        cv2.imwrite(str(output), scene_img)
        return detected_objects

    # Draw a bounding box and annotate the object name on the scene image
    def draw_box(self, scene_img, obj_name, H, img_dim, img_homography_inliers):
        h, w = img_dim['shape']
        obj_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        scene_corners = cv2.perspectiveTransform(obj_corners, H)

        # bounding box
        cv2.polylines(
            scene_img,
            [np.int32(scene_corners)], isClosed=True, color=(0, 255, 0), thickness=3
        )

        # annotation
        x, y = scene_corners[0][0]
        cv2.putText(scene_img, self.object_map[obj_name], (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(255, 0, 0), thickness=10)

    # finds homography using RANSAC
    def homography(self, obj_keypoints, keypts, good_matches, scene_img, obj_name, obj_data):

        obj_kps = np.float32([obj_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        scene_kps = np.float32([keypts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(obj_kps, scene_kps, cv2.RANSAC, 10.0)

        inlier_matches = []
        if mask is not None:
            for i, m in enumerate(good_matches):
                if mask[i][0] == 1:
                    inlier_matches.append(m)

        # Get matches of the inliers after homography (used to get a better boundary for the bounding box)
        img_homography_inliers = cv2.drawMatches(
            obj_data['image'], obj_data['keypoints'],
            scene_img, keypts,
            inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Draw the bounding box
        if len(inlier_matches) >= 23:
            self.draw_box(scene_img, obj_name, matrix, obj_data, img_homography_inliers)

        if matrix is not None:
            inliers = np.sum(mask)
            print(inliers)
            return inliers >= 23, matrix

        return False, None

    # Process all scenes, annotate and draw bounding boxes on them, and save results.
    def process_scenes(self):
        output_txt = self.home / "testing.txt"

        with open(output_txt, 'w') as f:
            for scene_view in ['front', 'left', 'right']:
                for num in range(1, 11):
                    scene = self.scenes / f"S{num}_{scene_view}.JPG"
                    detected_objects = self.detect_and_annotate(scene)


                    detected_objects2 = [f"{obj} ({self.object_map[obj]})" for obj in detected_objects]
                    results = f"{scene.name}: {', '.join(detected_objects2)}\n"

                    f.write(results)  # Write to the text file


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    detector = ObjectDetector(BASE_DIR)
    detector.load_objects()
    detector.process_scenes()
