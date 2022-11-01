Machine Learning applied to Galaxy Morphology (Barchi+ 2019)
===============================================================================
Machine and Deep Learning Applied to Galaxy Morphology - A Comparative Study
	Barchi P.H., de Carvalho R.R., Rosa R.R., Sautter R., Soares-Santos M., Marques B.A.D., Clua E., Gonçalves T.S., de Sá Freitas C., Moura T.C.
===============================================================================
Keywords: 
	galaxies: photometry; methods: data analysis; machine learning; techniques: image processing; galaxies: general; catalogs

Description:
	Machine and Deep Learning morphological classification for 670,560 galaxies from Sloan Digital Sky Survey Data Release 7 (SDSS-DR7). Classifications are provided for 2 classes problem (0: elliptical; or, 1: spiral galaxy) and 3 classes problem (0: elliptical, 1: non-barred spiral, or 2: barred spiral galaxy). ML2classes classification is obtained by Traditional Machine Learning Approach, using Morphological non-parametric parameters and Decision Tree. Classifications using Deep Learning are obtained using a Convolutional Neural Network (CNN). Morphological non-parametric parameters are provided as well: Concentration (C), Asymmetry (A), Smoothness (S), Gradient Pattern Analysis (G2) parameter and Entropy (H). We also provide the Error from CyMorph processing. All error flags are mapped as follows: Error = 0: success (no errors); Error = 1: many objects of significant brightness inside 2 Rp of the galaxy; Error = 2: not possible to calculate the galaxy's Rp; Error = 3: problem calculating GPA; Error = 4: problem calculating H; Error = 5: problem calculating C; Error = 6: problem calculating A; Error = 7: problem calculating S.
===============================================================================
Description of columns
<column number>	<column name>	= <description>
-------------------------------------------------------------------------------
1	dr7objid				= SDSS DR7 Id
2	TType					= please see Dominguez-Sanchez et al. (2018)
3	K 	 					= Area of the galaxy’s Petrosian ellipse divided by the area of the of the PSF, estimated by the Full Width at Half Maximum (FWHM)
4	C						= Concentration
5	A						= Asymmetry
6	S						= Smoothness
7 	G2						= Second gradient moment from Gradient Pattern Analysis (GPA)
8	H						= Entropy
9 	Error					= CyMorph Error flag
10	ML2classes				= Classification obtained by Traditional Machine Learning Approach (Decision Tree)
11	CNN2classes1stClass 	= Class with the highest probability obtained by CNN considering the 2 classes problem
12	CNN2classes1stClassPerc	= Probability percentage of the 1st class obtained by CNN considering the 2 classes problem
13	CNN2classes2ndClass		= Class with the lowest probability obtained by CNN considering the 2 classes problem
14	CNN2Classes2ndClassPerc	= Probability percentage of the 2nd class obtained by CNN considering the 2 classes problem
15	CNN3classes1stClass 	= Class with the highest probability obtained by CNN considering the 3 classes problem
16	CNN3Classes1stClassPerc	= Probability percentage of the 1st class obtained by CNN considering the 3 classes problem
17	CNN3Classes2ndClass 	= Class with the 2nd highest probability obtained by CNN considering the 3 classes problem
18	CNN3Classes2ndClassPerc	= Probability percentage of the 2nd class obtained by CNN considering the 3 classes problem
19	CNN3Classes3rdClass		= Class with the lowest probability obtained by CNN considering the 3 classes problem
20	CNN3Classes3rdClassPerc	= Probability percentage of the 3rd class obtained by CNN considering the 3 classes problem
=============================================================================== 