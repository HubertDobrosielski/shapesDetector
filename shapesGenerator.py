from PIL import Image, ImageDraw, ImageShow
import os
import random
from tqdm import tqdm
import math
import random
import numpy as np
import cv2
import progressbar


X = 0
Y = 1
MIN_SHAPE_SIZE = 20


def applyPerspectiveTransformation(image):
    perspectiveMatrix = (1, 0.2, 50, 0.1, 1, 30, 0, 0)

    width, height = image.size
    transformed_image = image.transform(
        (width, height), Image.PERSPECTIVE, perspectiveMatrix
    )
    return transformed_image


def getRandomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def getRandomAngle():
    return random.randint(0, 360)


def createBackground(width, height):
    color = getRandomColor()
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    return image, draw


def pasteShape(image, shape):
    image.paste(
        shape,
        (
            random.randint(0, image.size[X] - shape.size[X]),
            random.randint(0, image.size[Y] - shape.size[Y]),
        ),
        shape,
    )
    return image


def addNoiseToImage(image):
    noise_intensity = random.randint(0, 225)
    if noise_intensity <= 0:
        return image

    # Konwersja obrazu PIL na tablicę numpy
    image_array = np.array(image)

    if image_array.shape[2] == 3:
        # Jeśli obraz nie ma kanału alfa, dodaj go wypełniony wartościami 255
        alpha_channel = np.full(image_array.shape[:2], 255, dtype=np.uint8)
        image_array = np.dstack([image_array, alpha_channel])

    # Generowanie szumu o rozmiarze elipsy
    noise = np.random.randint(0, noise_intensity, image_array.shape[:2], dtype=np.uint8)

    # Rozmiar szumu dostosowany do rozmiaru elipsy
    noise_image = Image.fromarray(
        np.dstack([noise, noise, noise, np.full(noise.shape, 255, dtype=np.uint8)]),
        "RGBA",
    )
    noise_image = noise_image.resize(image.size)

    # Dodawanie szumu tylko do kanałów RGB (bez kanału alfa)
    noisy_image_array = np.clip(
        image_array[..., :3] + np.array(noise_image)[:, :, :3], 0, 255
    )

    # Połącz obraz z kanałem alfa
    noisy_image = Image.fromarray(
        np.dstack([noisy_image_array, image_array[..., 3]]), "RGBA"
    )

    return noisy_image


def drawEllipse(image):
    fillColor = getRandomColor()

    majorAxis = random.randint(MIN_SHAPE_SIZE, int(image.size[X] * (3 / 4)))
    # majorAxis = int(image.size[X] * (3 / 4))
    minorAxis = random.randint(MIN_SHAPE_SIZE, majorAxis)
    # minorAxis = majorAxis


    center = (majorAxis * 5 // 2, minorAxis * 5 // 2)

    ellipseImage = Image.new(
        "RGBA", (majorAxis * 5 + 2, minorAxis * 5 + 2), (0, 0, 0, 0)
    )

    point1 = (center[X] - (majorAxis // 2), center[Y] - (minorAxis // 2))
    point2 = (center[X] + (majorAxis // 2), center[Y] + (minorAxis // 2))
    ImageDraw.Draw(ellipseImage).ellipse([point1, point2], fill=fillColor)

    angle = random.randint(0, 360)
    ellipseImage = ellipseImage.rotate(angle, expand=True)

    # ellipseImage = applyPerspectiveTransformation(ellipseImage)

    ellipseImage = ellipseImage.crop(ellipseImage.getbbox())

    return ellipseImage


def generateRandomRectangle(image):
    fillColor = getRandomColor()
    angle = getRandomAngle()
    shapeWidth = random.randint(MIN_SHAPE_SIZE, int(image.size[X] * (3 / 4)))
    shapeHeight = random.randint(MIN_SHAPE_SIZE, int(image.size[X] * (3 / 4)))

    diagonal = int(math.sqrt(shapeWidth**2 + shapeHeight**2)) + 10

    # backGroundSize = max(shapeWidth*5 + 2, shapeHeight*5 + 2)

    center = (diagonal // 2, diagonal // 2)
    rectangleImage = Image.new("RGBA", (diagonal*2, diagonal*3), (0, 0, 0, 0))

    point1 = (center[0] - (shapeWidth // 2), center[1] - (shapeHeight // 2))
    point2 = (center[0] + (shapeWidth // 2), center[1] + (shapeHeight // 2))

    ImageDraw.Draw(rectangleImage).rectangle([point1, point2], fill=fillColor)

    rectangleImage = rectangleImage.rotate(angle, expand=True)
    # rectangleImage = applyPerspectiveTransformation(rectangleImage)
    rectangleImage = rectangleImage.crop(rectangleImage.getbbox())

    return rectangleImage

from PIL import Image, ImageDraw
import random
import math

MIN_SHAPE_SIZE = 10  # Minimalny rozmiar trójkąta

def generateRandomTriangle(image):
    fillColor = getRandomColor()
    angle = getRandomAngle()
    
    # Losowe boki trójkąta w zakresie od MIN_SHAPE_SIZE do image.size[X] * (3 / 4)
    side1 = random.randint(MIN_SHAPE_SIZE, int(image.size[X] * (3 / 4)))
    side2 = random.randint(MIN_SHAPE_SIZE, int(image.size[X] * (3 / 4)))
    side3 = random.randint(MIN_SHAPE_SIZE, int(image.size[X] * (3 / 4)))

    # Wysokość trójkąta obliczamy jako (pierwiastek(3)/2) * bok
    shapeHeight = int((math.sqrt(3) / 2) * max(side1, side2, side3))

    diagonal = int(math.sqrt(shapeHeight**2 + (side1 + side2 + side3)**2)) + 10

    center = (diagonal // 2, diagonal // 2)
    triangleImage = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))

    # Współrzędne wierzchołków trójkąta
    point1 = (center[0], center[1] - (shapeHeight // 2))
    point2 = (center[0] - (side1 // 2), center[1] + (shapeHeight // 2))
    point3 = (center[0] + (side2 // 2), center[1] + (shapeHeight // 2))

    ImageDraw.Draw(triangleImage).polygon([point1, point2, point3], fill=fillColor)

    triangleImage = triangleImage.rotate(angle, expand=True)
    triangleImage = triangleImage.crop(triangleImage.getbbox())


    return triangleImage



# MAIN ############################################################################

A = 500
GENERATE_SIZE = (A, A)
SAVE_SIZE = (A, A)
shapeName = "ellipse"


sizeList = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
shapeNamesList = ["ellipse", "rectangle","triangle","none"]
numberOfSamples = 100



# Oblicz całkowitą liczbę iteracji
total_iterations = len(sizeList) * len(shapeNamesList) * numberOfSamples
# Utwórz pasek postępu
bar = progressbar.ProgressBar(max_value=total_iterations)


for size in sizeList:
    for shapeName in shapeNamesList:
        for sample in range(numberOfSamples):

            image, draw = createBackground(size, size)

            shape = None
            if shapeName == "ellipse":
                shape = drawEllipse(image)
            elif shapeName == "rectangle":
                shape = generateRandomRectangle(image)
            elif shapeName == "triangle":
                shape = generateRandomTriangle(image)

            if shape != None:
                image = pasteShape(image, shape)

            image = addNoiseToImage(image)
            # image = image.resize((saveWidth,saveHeigh))
            # shape.save(os.path.join("ob.png"))
            image.save(
                os.path.join(f"photosX/{shapeName}/{shapeName}_{size}_{sample}.png")
            )

            # Aktualizuj pasek postępu
            bar.update(bar.value + 1)