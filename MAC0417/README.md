# ndarray

course: **pirp_5e PIRP Python/Numpy**

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

* notation:

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

* binary image from above:

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

a.reshape(row, column):

        a = [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

        a.reshape(4,5) = 
        [[ 0  1  2  3  4]
        [ 5  6  7  8  9]
        [10 11 12 13 14]
        [15 16 17 18 19]]

* fatiamento 2D:

        a[1,:]  # 2nd row
        a[:,0]  # 1st column

        a[0::2,:] # access row 2 by 2, starting from row 0
        > [[ 0  1  2  3  4]
           [10 11 12 13 14]]

        a[0::2,1::2] # access row and column 2 by 2, starting from row 0, column 1
        > [[ 1  3]
           [11 13]]

        a[a::r, b::c] # from line a, jump lines by r, and from column b, jump column by c

        a[2::1, 3::1]
        > [[13, 14],
           [18, 19]]

        a[-1:-3:-1,:]
        > [[15 16 17 18 19]
           [10 11 12 13 14]]

        a[::-1,:]
        > [[15 16 17 18 19]
           [10 11 12 13 14]
           [ 5  6  7  8  9]
           [ 0  1  2  3  4]]

* Inside "Fatiamento no ndarray bidimensional":
    * [Processamento de imagens usando fatiamento do Numpy](http://adessowiki.fee.unicamp.br/adesso-1/wiki/master/tutorial_1_imagens/view/)
    * [iaprofiledemo - Extraction and plotting of a vertical profile of an image](http://adessowiki.fee.unicamp.br/adesso-1/wiki/ia636/iaprofiledemo/view/)

* Amplify image:

        a.shape # we can verify the a's shape


        import numpy as np

        f = adreadgray('gear.tif')
        adshow(f, 'original %s' % (f.shape,) )
        H,W = f.shape                               # H, W receives a pair of results
        g = np.zeros( (2*H,2*W), 'uint8')

        # the sections receives 
        g[ ::2, ::2] = f
        g[1::2, ::2] = f
        g[1::2,1::2] = f
        g[ ::2,1::2] = f
        adshow(g, 'ampliada por replicação %s' % (g.shape,) )


```python

    import numpy as np

    a = np.arange(25)
    a = a.reshape(5,5)

    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    H, W = a.shape
    g = np.zeros((2*H,2*W), int)

    g[ ::2, ::2] = a

        array([[ 0,  0,  1,  0,  2,  0,  3,  0,  4,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 5,  0,  6,  0,  7,  0,  8,  0,  9,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [10,  0, 11,  0, 12,  0, 13,  0, 14,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [15,  0, 16,  0, 17,  0, 18,  0, 19,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [20,  0, 21,  0, 22,  0, 23,  0, 24,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])

    g[1::2, ::2] = a

        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  1,  0,  2,  0,  3,  0,  4,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 5,  0,  6,  0,  7,  0,  8,  0,  9,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [10,  0, 11,  0, 12,  0, 13,  0, 14,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [15,  0, 16,  0, 17,  0, 18,  0, 19,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [20,  0, 21,  0, 22,  0, 23,  0, 24,  0]])


        # so far...

            array([[ 0,  0,  1,  0,  2,  0,  3,  0,  4,  0],
                   [ 0,  0,  1,  0,  2,  0,  3,  0,  4,  0],
                   [ 5,  0,  6,  0,  7,  0,  8,  0,  9,  0],
                   [ 5,  0,  6,  0,  7,  0,  8,  0,  9,  0],
                   [10,  0, 11,  0, 12,  0, 13,  0, 14,  0],
                   [10,  0, 11,  0, 12,  0, 13,  0, 14,  0],
                   [15,  0, 16,  0, 17,  0, 18,  0, 19,  0],
                   [15,  0, 16,  0, 17,  0, 18,  0, 19,  0],
                   [20,  0, 21,  0, 22,  0, 23,  0, 24,  0],
                   [20,  0, 21,  0, 22,  0, 23,  0, 24,  0]])

    g[1::2,1::2] = a

        # so ...

        array([[ 0,  0,  1,  0,  2,  0,  3,  0,  4,  0],
               [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4],
               [ 5,  0,  6,  0,  7,  0,  8,  0,  9,  0],
               [ 5,  5,  6,  6,  7,  7,  8,  8,  9,  9],
               [10,  0, 11,  0, 12,  0, 13,  0, 14,  0],
               [10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
               [15,  0, 16,  0, 17,  0, 18,  0, 19,  0],
               [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
               [20,  0, 21,  0, 22,  0, 23,  0, 24,  0],
               [20, 20, 21, 21, 22, 22, 23, 23, 24, 24]])


    g[ ::2,1::2] = a

        # so we have...

        array([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4],
               [ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4],
               [ 5,  5,  6,  6,  7,  7,  8,  8,  9,  9],
               [ 5,  5,  6,  6,  7,  7,  8,  8,  9,  9],
               [10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
               [10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
               [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
               [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
               [20, 20, 21, 21, 22, 22, 23, 23, 24, 24],
               [20, 20, 21, 21, 22, 22, 23, 23, 24, 24]])

```

* Sem Copia: 'a' is a np array. Doing b = a makes 'b' to share the same memory from 'a'. Modifying the 'b' content modifies also 'a'. To check that: `np.may_share_memory(a,b)`

* Copia rasa: most used method.
  * Reshape:

```python

a = np.arange(30)
b = a.reshape( (5, 6))
b[:, 0] = -1               # this line modifies 'a' and 'b' contents

c = a.reshape( (2, 3, 5) ) # we are having 2 matrices (3,5)

'c.base is a:',c.base is a # gives us TRUE
print 'np.may_share_memory(a,c):',np.may_share_memory(a,c) # gives us TRUE

```

  * Slice:

```python

import ia636 as ia

a = np.zeros( (5, 6))
b = a[::2,::2]        # **note: 'b' receives the even indexes from 'a', becoming (3,3)

b[:,:] = 1            # inserting 1 in all positions
print 'b.base is a:',b.base is a      # True
print 'np.may_share_memory(a,b):',np.may_share_memory(a,b)  # True

# other process

a = np.arange(25).reshape((5,5))
b = a[:,0]   # 'b' receives 1st column from 'a'

b[:] = np.arange(5) # **note: we can't do "b = np.arrange(5), otherwise, a new variable is created"

```

  * Transposto:

```python

a = np.arange(24).reshape((4,6))
at = a.T

print 'np.may_share_memory(a,at):',np.may_share_memory(a,at)    # True, so modiying 'at' modifies 'a' too


# this produces:

a:
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
at:
[[ 0  6 12 18]
 [ 1  7 13 19]
 [ 2  8 14 20]
 [ 3  9 15 21]
 [ 4 10 16 22]
 [ 5 11 17 23]]

```

  * Ravel:

```python

a = np.arange(24).reshape((4,6))
av = a.ravel()      # linear view of 'a'

print 'np.may_share_memory(a,av):',np.may_share_memory(a,av) # True

# produces:

a:
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
av.shape: (24,)
av:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]

```

* Copia profunda: produces different memories locations. Verifying ids such as `id(a)` will do the job. There are 2 ways to copy: through `copy` and `nparray( , copy=True)`:

```python

b = a.copy()  # 1st method
c = np.array(a, copy=True) # 2nd method

```

## Operacoes matriciais

```python

import numpy as np

a = np.arange(20).reshape(5,4)
b = 2 * np.ones((5,4))                # fill with 1s
c = np.arange(12,0,-1).reshape(4,3)   # from 12 to 0, increment -1, and reshape it

a=
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]]
b=
[[ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]
 [ 2.  2.  2.  2.]]
c=
[[12 11 10]
 [ 9  8  7]
 [ 6  5  4]
 [ 3  2  1]]

# multiplicacao
b5 = 5 * b # will multiplies every element from 'b' with 5
# soma
amb = a + b
# multiplicacao entre matrizes: dot()
ac = a.dot(c) # condition: columns from 'a' == rows from 'c'



```


## Quiz

### Função de geração de imagem com quadrados cinzas

```python

# Minha solução

 1 def qc( isImg):
 2     """ Gera imagem com quadrados cinzas
 3     isImg: se verdadeiro, retorna imagem 300 x 600
 4            se falso, retorna imagem 6 x 12
 5     """
 6     import numpy as np
 7     H,W = 300,600
 8     if not isImg:
 9         H = H/50
10         W = W/50
11     # edite aqui
12     f = np.zeros((H,W))
13 
14     f[:,:W/2] = 64
15     f[:,W/2:] = 192
16     f[H/3:2*H/3, W/6:W/3] = 128
17     f[H/3:2*H/3, 2*W/3:5*W/6] = 128
18 
19     return f

# Solução mais eficiente

01. def qc_slice( isImg ):
02.     import numpy as np
03.     H,W = 300,600
04.     if not isImg:
05.         H /= 50
06.         W /= 50
07.     f = np.empty((H,W), "uint8")
08.     f[:,:W/2] = 64
09.     f[:,W/2:] = 192
10.     f[H/3:2*H/3,W/6:2*W/6] = 128
11.     f[H/3:2*H/3,2*W/3:5*W/6] = 128
12.     return f

```

So... Yeah! **Success**!


