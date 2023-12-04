Scheme
======
		 ---------------
 -------------  |C31   C26   C43|
|C36  C25  C29| |		|
|	      | |		|
|C42  C46  C30| |C45   C32   C38|
 -------------   ---------------
--C133---C134--

-C42Finf: frame240, frame400, frame560, frame710, frame860, frame1011, frame1070, frame1430, frame1600, frame1630, frame1800, frame1890, frame2120
-C43Finf: frame20, frame540, frame850, frame1140, frame1280, frame1570, frame1690, frame1770, frame1850, frame1970, frame2180, frame2230, frame2290, frame2300, frame2310, frame2320
-C46Finf: frame100, frame170, frame310, frame570, frame1260, frame1420, frame1830*, frame2100
-C47Finf: frame70, frame560*, frame590, frame680, frame710*, frame1190, frame1830, frame2030
-C51Finf: frame240*, frame250, frame280*, frame330*, frame430*, frame440, frame450, frame570*, frame600, frame660, frame680*, frame770, frame1180, frame1210, frame1430, frame1480, frame1490, frame1730, frame2140
-C52Finf: frame270, frame280, frame330, frame430, frame1400, frame1650, frame1700, frame1870, frame1890*
-C67Finf: frame40, frame60, frame80, frame160, frame200, frame210, frame350, frame370, frame1080, frame1690*, frame1780, frame1900, frame1940, frame2000
-C74Finf: frame10
-C78Finf: frame0, frame230

*: changed name in the final folder due to overlapping with a frame from another video

-C46 y C50 funcionan bien completamente con la misma configuración
-C51 funciona bien hasta frame 2035 (de 2145) 94.8%
-C52 funciona bien hasta frame 625 (de 2715) 23.0%
-C47 funciona bien hasta frame 500 (de 3000) 16.6%
-C67 funciona bien hasta frame 1506 (de 2055), 73.2%

Update (nuevo método para tomar targets y diferenciar con codetargets):
-C46 bajó a 45%
-C47 mejoró a 53%
-C50 funciona bien
-C51 bajó a 58%
-C52 sigue igual en 23%
-C67 bajó a 59%

Update2 (en línea: 264 cambiar len(contours) > 3 a len(contours) > 2)
-C46 sigue en 45%
-C47 bajó a 52%
-C50 funciona bien
-C51 sigue en 58%
-C52 sigue igual en 23%
-C67 mejoró a 74%

Update3 (volver al método de identificación anterior más los nuevos codetarget) (agregar CSB targets)
-C46 funciona bien
-C47 mejoró a 57% (funciona en 2 partes)
-C50 funciona bien
-C51 funciona bien
-C52 sigue igual en 23% mejor se elimina por varios cachos
-C67 funciona bien
Se agregaron C42 y C43
-C42 funciona bien
-C43 funciona bien
