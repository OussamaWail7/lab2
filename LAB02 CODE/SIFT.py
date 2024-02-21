import numpy as np
import imutils
import cv2
import argparse

class Stitcher:
    def __init__(self):
        # Déterminer si nous utilisons OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # Déballer les images, puis détecter les points clés et extraire
        # les descripteurs locaux invariants
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Faire correspondre les descripteurs entre les deux images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # Si le match est None, alors il n'y a pas assez de points correspondants
        # pour créer un panorama
        if M is None:
            return None

        # Appliquer une transformation de perspective pour assembler les images
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # Vérifier si les correspondances de points clés doivent être visualisées
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # Retourner un tuple de l'image panoramique et de la visualisation
            return (result, vis)

        # Retourner l'image panoramique
        return result

    def detectAndDescribe(self, image):
        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Vérifier si nous utilisons OpenCV 3.X
        if self.isv3:
            # Détecter et extraire des caractéristiques de l'image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        else:
            # Détecter des points clés dans l'image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            # Extraire des caractéristiques de l'image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # Convertir les points clés des objets KeyPoint en tableaux NumPy
        kps = np.float32([kp.pt for kp in kps])

        # Retourner un tuple de points clés et de caractéristiques
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # Calculer les correspondances brutes et initialiser la liste des correspondances réelles
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # Boucle sur les correspondances brutes
        for m in rawMatches:
            # S'assurer que la distance est dans un certain rapport l'une de l'autre (test du rapport de Lowe)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Calculer une homographie entre les deux ensembles de points
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        # Aucune homographie ne peut être calculée
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True,
        help="path to the first image")
    ap.add_argument("-s", "--second", required=True,
        help="path to the second image")
    args = vars(ap.parse_args())

    # charger les deux images et les redimensionner pour avoir une largeur de 400 pixels
    imageA = cv2.imread(args["first"])
    imageB = cv2.imread(args["second"])
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)

    # assembler les images pour créer un panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    # afficher les images
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
