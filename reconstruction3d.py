import cv2
import numpy as np
import matplotlib.pyplot as plt

# focale en pixels (fx, fy) et point principal (cx, cy)
K = np.array([
    [1.69365276e+03, 0.0, 1.55551750e+03],
    [0.0, 1.54943655e+03, 2.03532072e+03],
    [0.0, 0.0, 1.0]
])

img1 = cv2.imread("IMG_1.JPG")
img2 = cv2.imread("IMG_2.JPG")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Keypoints : positions, orientation, échelle
sift = cv2.SIFT_create()

# keypoints + descripteurs de l'image 1
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# k=2 retourne les deux meilleurs matches pour chaque point de img1
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []
good_matches = []

# ratio de lowe
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)


img_matches = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches[:100],  # limiter pour lisibilité
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(15,6))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Matches SIFT (après ratio test)")
plt.axis("off")
plt.show()

#  matrice fondamentale qui fait le plan epipolaire entre les 2 images
#p2ᵀ · F · p1 = 0
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# translation entre les 2 cameras 
#E = Kᵀ · F · K
E, _ = cv2.findEssentialMat(
    pts1, pts2, K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

# r pour rotation , t pour translation e pour essentielle
_, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, K)


# matrice de reconstruction p1 et p2 a travers la metrice intrinseque k
P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K @ np.hstack((R, T))


# triangularisation vec p1 et p2
pts1_T = pts1.T.astype(np.float32)
pts2_T = pts2.T.astype(np.float32)

# # passage en coordonnées cartésiennes x,y,z
points_4D = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)
points_3D = points_4D[:3] / points_4D[3]
points_3D = points_3D.T



colors = []
for pt in pts1:
    x, y = int(pt[0]), int(pt[1])
    colors.append(img1[y, x][::-1] / 255.0)

colors = np.array(colors)

# on garde que le  points avec la profondeur z 
mask_valid = (points_3D[:,2] > 0) & (points_3D[:,2] < 5)
points_3D = points_3D[mask_valid]
colors = colors[mask_valid]



fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_3D[:,0], points_3D[:,1], points_3D[:,2],
           c=colors, s=2)

ax.set_title("Reconstruction 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

np.savetxt("points_3D.xyz", points_3D)
print("Points reconstruits :", len(points_3D))

print("Vecteur translation T :", T)
print("Matrice rotation R :\n", R)