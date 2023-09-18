import cv2

print("Processing....")
print("Masking done")
print("Image restored succesfully.")
image = cv2.imread('./dataset/damaged/57.jpg')
mask = cv2.imread('./dataset/masks/57.jpg')

flags = cv2.INPAINT_TELEA
#flags = cv2.INPAINT_NS


mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#inpainting
output = cv2.inpaint(image, mask, 1, flags=flags)


cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
