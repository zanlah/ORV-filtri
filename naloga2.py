import cv2 as cv
import numpy as np

def konvolucija(slika, jedro):
    #todo

    #dobim dimenzije slike in jedra
    visina, sirina = slika.shape
    visina_j, sirina_j = jedro.shape

    #ugotovim kako je potrebno razširiti sliko da se ne bo zmanjšala po končanju konvolucije
    razširitev_v = visina_j // 2
    razširitev_š = sirina_j // 2

    nova_visina = visina + 2 * razširitev_v
    nova_sirina = sirina + 2 * razširitev_š

    #razširjena_slika = np.zeros((nova_visina, nova_sirina), dtype=np.float32)

    # Ponovi robne piksle za razširitev
    # Zgornji in spodnji rob
    #for i in range(razširitev_š, sirina + razširitev_š):
     #   razširjena_slika[:razširitev_v, i] = slika[0, i-razširitev_š]
      #  razširjena_slika[visina+razširitev_v:, i] = slika[-1, i-razširitev_š]

    # Levi in desni rob
    #for i in range(nova_visina):
     #   razširjena_slika[i, :razširitev_š] = razširjena_slika[i, razširitev_š]
      #  razširjena_slika[i, sirina+razširitev_š:] = razširjena_slika[i, sirina+razširitev_š-1]
    #razširim robbne piksle
    #razširjena_slika = np.pad(slika, ((razširitev_v, razširitev_v), (razširitev_š, razširitev_š)), 'edge')

    #dodam ničle okoli robov
    razširjena_slika = np.pad(slika, ((razširitev_v, razširitev_v), (razširitev_š, razširitev_š)), 'constant', constant_values=0)
    rezultat = np.zeros((visina, sirina), dtype=np.float32)
    for i in range(visina):
        for j in range(sirina):
            # Izračunaj uteženo vsoto med jedrom in ustreznim delom razširjene slike
            območje = razširjena_slika[i:i+visina_j, j:j+sirina_j]
            rezultat[i, j] = np.sum(območje * jedro)

    #rezultat = cv.normalize(rezultat, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

    return rezultat

    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.''' 
    pass

def filtriraj_z_gaussovim_jedrom(slika,sigma):
    velikost_jedra = int((2*sigma)*2 + 1)

    #srednji index jedra
    k = (velikost_jedra - 1) // 2
    
    jedro = np.zeros((velikost_jedra, velikost_jedra), dtype=np.float32)

    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            x = i - k
            y = j - k
            jedro[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalizacija jedra - vsota vseh elementov je enaka 1        
    jedro /= jedro.sum()

    return konvolucija(slika, jedro)


def filtriraj_sobel_smer(slika):
     # Sobelova jedra
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_jedro = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    return konvolucija(slika, sobel_jedro)


if __name__ == '__main__':  
   
    
    
    ############################################
    #Prva slika
    slika1 = np.array([[1, 0, 0, 0],
                   [0, 2, 0, 0],
                   [0, 0, 3, 0],
                   [0, 0, 0, 4]], dtype=np.float32)

    ###########
    # 1 0 0 0 #
    # 0 2 0 0 #
    # 0 0 3 0 #
    # 0 0 0 4 #
    ###########

    jedro = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])
    
    slika_konvolucija = konvolucija(slika1, jedro)
    slika_gauss = filtriraj_z_gaussovim_jedrom(slika1, 1.4)
    slika_sobel = filtriraj_sobel_smer(slika1)
    # Prikažemo izvirno sliko
    print("Original:\n {}".format(slika1))
    print("Konvolucija:\n {}".format(slika_konvolucija))
    print("Gauss:\n {}".format(slika_gauss))
    print("Sobel:\n {}".format(slika_sobel))
    print("######################")
    
#    slika = cv.imread(".utils/lenna.png", cv.IMREAD_COLOR)  
#    if slika is not None:
#
#        kanali = cv.split(slika)
#        
#        # Apply the Gaussian filter to each channel
#        filtrirani_kanali = [filtriraj_z_gaussovim_jedrom(kanal, 1) for kanal in kanali]
#        
#        # Merge the channels back into an RGB image
#        filtrirana_slika = cv.merge(filtrirani_kanali)
#
#        # Prikaži originalno in filtrirano sliko
#        cv.imshow('Originalna Slika', slika)
#        cv.imshow('Filtrirana Slika', filtrirana_slika)
#        cv.waitKey(0)
#        cv.destroyAllWindows()
   
