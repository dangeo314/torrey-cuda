#include "texture.cuh"
#include "flexception.h"

__host__ __device__ Vector3 eval(const Texture &texture, const Vector2 &uv) {
    if (texture.type == CONSTANT) {
        const ConstantTexture* constant = &texture.data.constant;
        return constant->color;
    /*
    } else if (auto *image = std::get_if<ImageTexture>(&texture)) {
        const Image3 &img = image->image;
        Real u = modulo(image->uscale * uv[0] + image->uoffset, Real(1)) *
            img.width;
        Real v = modulo(image->vscale * uv[1] + image->voffset, Real(1)) *
            img.height;
        int ufi = modulo(int(u), img.width);
        int vfi = modulo(int(v), img.height);
        int uci = modulo(ufi + 1, img.width);
        int vci = modulo(vfi + 1, img.height);
        Real u_off = u - ufi;
        Real v_off = v - vfi;
        Vector3 val_ff = img(ufi, vfi);
        Vector3 val_fc = img(ufi, vci);
        Vector3 val_cf = img(uci, vfi);
        Vector3 val_cc = img(uci, vci);
        return val_ff * (1 - u_off) * (1 - v_off) +
               val_fc * (1 - u_off) *      v_off +
               val_cf *      u_off  * (1 - v_off) +
               val_cc *      u_off  *      v_off;
    */
    } else {
        //Error("Unhandled Texture type");
        return Vector3{0, 0, 0};
    }
}
