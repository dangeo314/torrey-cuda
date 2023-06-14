HW3 BONUS:
    The bonus for Perlin Noise is shown in hw3_5
    The bonus for stratified sampling is also shown in hw3_5, but specifically in the new function called "radiance_perlin"

    Fresnel Argument Bonus: 
    The fresnel term has been a subject of controversy, becuase it's very difficult to get both accurate and artist-friendly/easy to use terms at the same time.
    usually, the industries adopt a compromise in between the two, however it won't always fix every edge case. For example, with the schlick approximation, we can find that
    the dielectric materials are respresented somewhere between alluminum and palladium, and have a deltaE that is reasonable, which I agree with the author on.
    However, chromium does not fit the real model very well in this case, due to the complexity of it's molecular structure. I don't believe that Chromium is the only example of this, so
    I think it's interesting that the author chose to include only one example that looked very bad, when in reality, there would probably be plenty more materials found in production that
    would be similar to the effects of the chromium approximation.


HW4:
    For my scene, I modified the lighting of the background and the area light of the Cornell Box scene to create a filter-like effect

HW4 BONUS:
    Russian Roulette Pathtracing implemented in hw4_3. The Russian Roulette pathtracing is great because it reduces computation time, while
    still maintaining a relatively accurate result. It is also an unbiased technique.