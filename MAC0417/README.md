# ndarray

Array with n-dimensions, indexed by a tuple with n integers.

* dtype - data type
* shape - tuple indicating dimensions

example:

    a = np.array( [2,3,4,-1,-2] ) # one row, five columns
    a.shape = 5
    a.dtype = int64

    b = np.array( [ [1.5, 2.3, 5.2],
            [4.2, 5.6, 4.4] ] )
    b.shape = (2, 3)
    b.shape[-1] = 3 # columns
    b-shape[-2] = 2 # rows

    > Em matrix unidimensional:
    > shape[0] indica numero de elementos -> colunas

    > Em matrix bidimensional:
    > ultimo elemento da tupla shape indica: numero de colunas (shape[-2])
    > penultimo, a linha (shape[-1]) ou (shape[0])

### Initialize

    d = np.zeros( (2,4) )                    # matrix com 2 rows e 4 columns, contendo 0s
    d = np.ones( (3,2,5), dtype='int16' )    # contem 1s
    d = np.empty( (2,3), 'bool' )            # contem False

### vetores sequencias

    np.arange( 10) =  [0 1 2 3 4 5 6 7 8 9]
    np.arange( 3, 8) =  [3 4 5 6 7]
    np.arange( 0, 2, 0.5) =  [ 0.   0.5  1.   1.5]         # 0 <= x < 2 with +0.5
    np.linspace( 0, 2, 5 ) =  [ 0.   0.5  1.   1.5  2. ]    # 0 <= x <= 2 with (5) as the last arg = number of elements


| dtype | valores |
|:-----:|:-------:|
| bool  | True, False                           |
| uint8 |8 bits sem sinal, de 0 a 255           |
| uint16 | 16 bits sem sinal, de 0 a 65535      |
| int     | 64 bits com sinal                   |
| float    | ponto flutuante                    |

### Read image

    f = adreadgray('cookies.tif')
    f.shape = (174, 314)
    f.dtype = uint8
    f.size = 54636 # total number of pixels
    print f
    [[24 32 32 ..., 49 41 41]
     [32 32 24 ..., 49 49 41]
     [32 32 32 ..., 49 49 41]
     ..., 
     [32 32 32 ..., 32 32 32]
     [32 32 32 ..., 32 32 32]
     [32 24 32 ..., 32 24 41]]

* notation

    0   -> black
    255 -> white

* show image: adshow(f,'Imagem cookies.tif')

    f_bin = f > 128             # f_bin[i, j] will contain 0 or 1, depending on if( f[i, j] > 128 )
    print 'Tipo do pixel:', f_bin.dtype
    > bool
    adshow(f_bin,'Imagem binária')


* colored images

    f_cor = adread('boat.tif')
    print 'Dimensões:    ', f_cor.shape
    print 'Tipo do pixel:', f_cor.dtype
    adshow(f_cor, 'Imagem colorida')
    
    > Dimensões:     (3, 257, 256)     # shape = (3, H, W) where 3 is 3 dimensions
    > Tipo do pixel: uint8

* printing grad. B to W:

    import numpy as np
    import ia636 as ia

    f = uint8(np.arange(256))
    f = np.tile(f, (128,1))
    print ia.iaimginfo(f)
    adshow(f,'256 gray tones')

* binary image from above

    g = f > 128
    print ia.iaimginfo(g)
    adshow(g,'binary image')
    gp = np.pad(g, ((1,1),(1,1)), 'constant', constant_values=((0,0),(0,0)))
    print ia.iaimginfo(gp)
    adshow(gp,'framed binary image')

** Note: adding subtitles:

    f = adreadgray('astablet.tif')
    H,W = f.shape
    legenda = 'astablet dimensões: (%d,%d)' % (H,W)
    adshow(f, legenda)

* Function iaimginfo
 
Print image size and pixel data type information. **iaimginfo** gives a string with the image size and pixel data type. The string also gives the minimum and maximum values of the image.

    from ia636 import iaimginfo

    a = adread('danaus.tif')
    print 'a:',iaimginfo(a)
    a_bin = a > 128
    print 'a_bin:', iaimginfo(a_bin)
    a_f = a * 3.5
    print 'a_f:', iaimginfo(a_f)

    > a: <type 'numpy.ndarray'> (3, 256, 256) uint8 0 255
    > a_bin: <type 'numpy.ndarray'> (3, 256, 256) bool False True
    > a_f: <type 'numpy.ndarray'> (3, 256, 256) float64 0.000000 892.500000

    import ia636
    print 'f:    ', ia636.iaimginfo(f)
    print 'f_bin:', ia636.iaimginfo(f_bin)
    print 'f_cor:', ia636.iaimginfo(f_cor)

    > f:     <type 'numpy.ndarray'> (174, 314) uint8 16 205        #  tipo de pixel, dimensões e os valores mínimo e máximo da imagem
    > f_bin: <type 'numpy.ndarray'> (174, 314) bool False True
    > f_cor: <type 'numpy.ndarray'> (3, 257, 256) uint8 0 218

* crop interest area from image

    f= adreadgray('cookies.tif')
    g = f[:7,:10]

* fatiamento

a[1:15:2], 1st arg = starting number, 2nd arg = end number, 3rd arg = 1st + 3rd arg

    a = np.arange(20) # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    a[1:15:2]

    > [ 1  3  5  7  9 11 13]

    a[1:-1:2] # fatiamento termina antes do ultimo elemento 

    > [ 1  3  5  7  9 11 13 15 17]

    a[-3:2:-1]

    > [17 16 15 14 13 12 11 10  9  8  7  6  5  4  3] # a[-3] = 17, until last print ( x > a[2] = 2 ), do i - 1


* fatiamento avancado

    a[:15:2] # starting from 0

    > [ 0  2  4  6  8 10 12 14]    

    a[1::2] # until the last element

    > [ 1  3  5  7  9 11 13 15 17 19]

    a[1:15] # step is + 1

    > [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]

    a[:] # from start to end, step + 1

    > [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

* reshaping

    a = [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    a.reshape(4,5) = 
    [[ 0  1  2  3  4]
    [ 5  6  7  8  9]
    [10 11 12 13 14]
    [15 16 17 18 19]]