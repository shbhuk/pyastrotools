'''
1) fnumber_convert()
2) i_to_r()
3) fresnel_loss()
4) radius_to_sag()
5) sag_to_radius()
6) beam_deviation()
7) beam_deviation_prism()
8) dispersion_sellmeier()
9) dispersion_cauchy()
10) dispersion_conrady()
11) absorption_loss_extrapolate
12) fp_etalon_transmission() - NEED TO CHECK
13) interference_dielectric() - NEED TO CHECK
14) interference_metallic() - NEED TO CHECK
15) fringe_frequency()
16) interference_amplitude()
17) attenuation_unitsconvert()
'''




import numpy as np
from astropy.io import fits
from scipy import misc
import numpy as np
import math
import re
import astropy.io
import astropy.constants as ac
from .astro_tools import change_spectral_bin
from astropy import units as u




def fnumber_convert(f=0,na=0,angle=0,n=1):
    '''
    Enter either f number, numerical aperture or cone angle and generate the other two values. Default refractive index is 1 (air)

    INPUT:
        f = f Number
        na = Numerical Aperture
        angle = Cone Angle in degrees. Half angle
        n = Refractive Index. Default = 1 for air

    '''

    if f!=0:
        F=f
        Angle=math.atan(1/(2.*f))
        NA=n*math.sin(Angle)

    if na!=0:
        NA=na
        Angle=math.asin(na/n)
        F=1/(2.*math.tan(Angle))

    if angle!=0:
        Angle=angle*np.pi/180
        NA=n*math.sin(Angle)
        F=1/(2.*math.tan(Angle))

    return F,NA,Angle*180/np.pi

def i_to_r(i,n1,n2 = 1):
    '''
    Find angle of refraction from angle of incidence using Snell's Law

    INPUT:
        i = Angle of Incidence [Degrees]
        n1 = Refractive Index of Substance refracting into
        n2 = Refractive Index (Generally Air)


    OUTPUT:
        r = Angle of refraction [Degrees

    '''

    i = i * np.pi/180.

    return np.arcsin(np.sin(i)*n2/n1)*180/np.pi





def fresnel_loss(i,n1,n2=1):
    '''
    Enter the two refractive indices and angle of incidence to obtain Fresnel losses.
    Assume unpolarized light. n2 -> n1
    INPUT:
        i = Angle if incidence in degrees

        n1,n2 = Refractive Indices of both materials. Default value for second refractive index is 1 (air)

    OUTPUT:
        Fresnel loss fraction
    '''

    if hasattr(n1, "__len__"):
        if isinstance(n2, (int,float)):
            n2 = np.repeat(n2, len(n1))
    else:
        n1 = np.array([n1])
        n2 = np.array([n2])




    rs = np.array([((n1[x]*math.cos(i*np.pi/180) - n2[x]*math.cos(i_to_r(i,n1[x],n2[x])*np.pi/180)) / (n1[x]*math.cos(i*np.pi/180) + n2[x]*math.cos(i_to_r(i,n1[x],n2[x])*np.pi/180)))**2 for x in range(0,len(n1))])

    rp = np.array([((-n2[x]*math.cos(i*np.pi/180) + n1[x]*math.cos(i_to_r(i,n1[x],n2[x])*np.pi/180)) / (n2[x]*math.cos(i*np.pi/180) + n1[x]*math.cos(i_to_r(i,n1[x],n2[x])*np.pi/180)))**2 for x in range(0,len(n1))])
    return (rs+rp)/2.


def radius_to_sag(r,d):
    '''
    Enter the radius of curvature and diameter to calculate sag of lens
    INPUT:
        r = Radius of Curvature [mm]
        d = Diameter [mm]
    OUTPUT:
        sag [mm]
    '''

    return r-math.sqrt((r*r)-(d/2.)**2)


def sag_to_radius(s,d):
    '''
    Enter the sag and diameter to calculate radius of curvature of lens
    INPUT:
        s = Sag [mm]
        d = Diameter [mm]
    OUTPUT:
        radius of curvature [mm]
    '''

    return (1./(2*s))*((s*s)+(d/2.)**2)


def beam_deviation(i,t,n=1.5):
    '''
    The deviation in a beam due to a plane parallel slab of n = 1.5 glass
    INPUT :
        i = Angle of Incidence in degrees
        t = Thickness of glass
        n = Refractive Index of Glass
    OUTPUT:
        Deviation in length units
    '''
    i = i * np.pi/180.
    r = math.asin(math.sin(i)/n)

    return t*math.sin(i-r)/math.cos(r)


def beam_deviation_prism(i,apex,n=1.5):
    '''
    Find the angle of deviation due to a prism

    INPUT:
        i = Angle of Incidence [degrees]
        apex = Apex Angle [degrees]
        n = Refractive Index
    OUTPUT:
        D = Deviation of the beam from incident beam [degrees]
    '''

    # Convert to Radians
    i = i * np.pi/180.
    apex = apex * np.pi/180

    # Beam deviation [radians]
    D = i - apex + math.asin(n*math.sin(apex - math.asin(math.sin(i)/n)))

    return D * 180./np.pi

def apex_from_deviation(D,n=1.5):
    '''
    Inverse of beam_deviation_prism()
    It calculates the apex angle given the angle of deviation assuming normal incidence

    INPUT:
        D = Angle of deviation from the apex [degrees]
        n = Refractive Index
    OUTPUT:
        apex = Apex angle of prism [degrees]

    '''

    D = D*np.pi/180

    return math.atan(math.sin(D)/(1.5-math.cos(D)))*180/np.pi




def dispersion_sellmeier(wl,b1,c1,b2,c2,b3,c3):
    '''
    Find the refractive index of a material using the Sellmeier formula

    INPUT:
        wl = List or array or scalar with wavelength [Angstroms]
        b1,c1,b2,c2,b3,c3 = Coefficients

    OUTPUT:
        n = Scalar or vector with refractive indices
    #  https://en.wikipedia.org/wiki/Sellmeier_equation
    '''

    wl = np.array(wl)
    wl = wl/10000.

    return np.sqrt((b1*wl*wl)/((wl*wl)-c1) + (b2*wl*wl)/((wl*wl)-c2) + (b3*wl*wl)/((wl*wl)-c3) + 1)


def dispersion_cauchy(wl,coeff):
    '''
    Find the refractive index of a material using the Cauchy Equation

    #       https://en.wikipedia.org/wiki/Cauchy%27s_equation
    INPUT:
        wl = List or array or scalar with wavelength. Wavelength units are generally determined by the coefficients.
        coeff = A list of coefficients for the Cauchy formula. Number of coefficients determines number of terms in Cauchy expression

        n(lambda) = coeff[0] + coeff[1]/lambda^2 + coeff[2]/lambda^4 + ...

    OUTPUT:
        n = Scalar or vector with refractive indices

    '''
    if isinstance(wl,(int,float)):
        wl = [wl]

    wl = np.array(wl)

    return [np.sum([coeff[x]/(float(wl[y])**(2*x)) for x in range(0,len(coeff))]) for y in range(0,len(wl))]

def dispersion_conrady(wl,coeff):
    '''
    Find the refractive index of a material using the Cauchy Equation

    # https://pdfs.semanticscholar.org/1fde/0c8d79ae3fb3564b85af92a3fc32648dde9f.pdf
    # https://customers.zemax.com/os/resources/learn/knowledgebase/fitting-index-data-in-zemax
    # https://www.gnu.org/software/goptical/manual/Material_Conrady_class_reference.html#__562

    INPUT:
        wl = List or array or scalar with wavelength [um]
        coeff = A list of coefficients for the Cauchy formula. Number of coefficients determines number of terms in Cauchy expression

        n(lambda) = coeff[0] + coeff[1]/lambda + coeff[2]/lambda^3.5

    OUTPUT:
        n = Scalar or vector with refractive indices

    '''
    if isinstance(wl,(int,float)):
        wl = [wl]

    wl = np.array(wl)

    return coeff[0]+coeff[1]/wl + coeff[2]/(wl**3.5)



def absorption_loss_extrapolate(x1,t,x2):
    '''
    Given the transmittance 't' of a material with thickness x1, what is the transmittance of material with thickness x2?

    INPUT :
        x1 - Thickness of test sample
        t - Transmittance of test sample 0 <= t <=1
        x2 - Thickness for which to check for

    OUTPUT :
        t2 - Transmittance of material with thickness x2

    '''

    return t * np.exp(np.log(t) * (x2 - x1)/x1)

def fp_etalon_transmission(aoi,R,n,d,wav):
    '''
    aoi - Angle of Incidence of the beam
    R - reflectivity of the surface
    n - Refractive index of cavity
    d - Width of the etalon (mm)
    wav - Wavelength of light (nm)

    Modern Optics by Guenther pg 103
    '''



    delta = 4*np.pi * d*1e-3 * n * np.cos(i_to_r(aoi, n1 = n, n2 = 1)*np.pi/180)/(wav*1e-9)
    F = 4 * (R/(1-R**2))**2
    T = 1./(1 + F*(np.sin(delta/2))**2)

    # delta = 4*np.pi*d*1e-3 * np.cos(aoi*np.pi/180)/(wav*1e-9)
    # offset = delta - np.floor(delta)
    # F = 4*R/(1-R)**2
    # T = 1./(1 + F*(np.sin(offset/2))**2)

    return T,F,delta

def _(r1_air2glass, r1_glass2air, r2_glass2air):
    """
    Calculate the interference fringe intensity modulation, especially if the coating is a dielectric and not metallic
    INPUTS:
        r1_air2glass = Reflection when going from Air to Glass on S1 (fraction)
        r1_glass2air = Reflection when going from Glass to Air on S1 (fraction). Since it is a dielectric,
                not necessarily = r1_air2glass
        r2_glass2air = Transmission when going from Glass to Air on S2 (fraction)

    OUTPUTS:
        fringe_peak2peak = Fringe peak to peak
        i1 = Intensity of light in primary
        i2 = Intensity of light in secondary

        Fringe amplitude as fraction of intensity = fringe_peak2peak/(i1)
    """

    i1 = (1-r1_air2glass) * (1-r2_glass2air)

    i2 = (1-r1_air2glass) * r2_glass2air * r1_glass2air * (1-r2_glass2air)

    fringe_peak2peak = 4 * np.sqrt(i1*i2)

    return fringe_peak2peak, i1, i2


def _(r1_air2glass, r2_glass2air):
    """
    Calculate the interference fringe intensity modulation, especially if the coating is metallic
    INPUTS:
        r1_air2glass = Reflection when going from Air to Glass on S1 (fraction)
        r2_glass2air = Transmission when going from Glass to Air on S2 (fraction)

    OUTPUTS:
        fringe_peak2peak = Fringe peak to peak
        i1 = Intensity of light in primary
        i2 = Intensity of light in secondary

        Fringe amplitude as fraction of intensity = fringe_peak2peak/(i1)
    """

    i1 = (1-r1_air2glass) * (1-r2_glass2air)

    i2 = (1-r1_air2glass) * r2_glass2air * r1_air2glass * (1-r2_glass2air)

    fringe_peak2peak = 4 * np.sqrt(i1*i2)

    return fringe_peak2peak, i1, i2



def fringe_frequency(L, n, wav):
    """
    INPUT:
        L = Thickness of the plate (mm)
        n = Refractive Index of the material
        wav = Wavelength (Angstroms)

    OUTPUT:


    """



    dnu = ac.c.value/(2*n*L*1e-3)

    dlambda = change_spectral_bin(dnu = dnu*u.Hz, wavelength = wav*u.AA)

    return dnu*u.Hz, dlambda['dlambda'].to(u.AA)


def interference_amplitude(R1, R2):
    """
    INPUTS:
        R1, R2: Reflectivity of surface S1 and S2 resp. (Fractions)
    OUTPUTS:
        Imax, Imin
        (Imax-Imin)/Imax - Peak to peak fringe normalized by maximum
    Holden et al. 1949
    """


    T1 = 1-R1
    T2 = 1-R2

    maxima = (T1*T2)/(1+R1*R2 - 2*np.sqrt(R1*R2))
    minima = (T1*T2)/(1+R1*R2 + 2*np.sqrt(R1*R2))


    return maxima,minima, (maxima-minima)/maxima

def attenuation_unitsconvert(Length, dBperkm=None):
    """
    Take the attenuation data in dB/km and convert it to absolute for given length    INPUTS:
        Length = Length of fiber [m]
        dBperkm = Attenuation value [dB/km]
    OUTPUTS:
        dBperkm
        Fractional
    """

    Fractional = (10**(-dBperkm/10))**(Length/1000)
    return Fractional
