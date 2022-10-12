import pygame
from pygame.locals import  *
from os.path import join

pygame.init()

window = pygame.display.set_mode((400,480))
screen = pygame.display.get_surface()


src_img = join("images","teens.png")
age = "Teens"

image = pygame.image.load(src_img)
image = image.convert()
image = pygame.transform.scale(image, (200,200))

police = pygame.font.SysFont("comicsansms", 74)
image_texte = police.render(age, 1, (0,0,255))
pygame.display.set_caption('Voice Age - Capgemini')

continuer = True

while continuer:
    screen.blit(image,(100,240))
    screen.blit(image_texte, (100,100))
    
    for event in pygame.event.get():
        if event.type == pygame.K_ESCAPE:
            continuer = False
    pygame.display.flip()

pygame.quit()
