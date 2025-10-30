// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unused)]
#![allow(clippy::type_complexity)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
use crate::*;
use jxl_simd::{F32SimdVec, SimdDescriptor};

#[allow(clippy::too_many_arguments)]
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub(super) fn idct_128<D: SimdDescriptor>(
    d: D,
    mut v0: D::F32Vec,
    mut v1: D::F32Vec,
    mut v2: D::F32Vec,
    mut v3: D::F32Vec,
    mut v4: D::F32Vec,
    mut v5: D::F32Vec,
    mut v6: D::F32Vec,
    mut v7: D::F32Vec,
    mut v8: D::F32Vec,
    mut v9: D::F32Vec,
    mut v10: D::F32Vec,
    mut v11: D::F32Vec,
    mut v12: D::F32Vec,
    mut v13: D::F32Vec,
    mut v14: D::F32Vec,
    mut v15: D::F32Vec,
    mut v16: D::F32Vec,
    mut v17: D::F32Vec,
    mut v18: D::F32Vec,
    mut v19: D::F32Vec,
    mut v20: D::F32Vec,
    mut v21: D::F32Vec,
    mut v22: D::F32Vec,
    mut v23: D::F32Vec,
    mut v24: D::F32Vec,
    mut v25: D::F32Vec,
    mut v26: D::F32Vec,
    mut v27: D::F32Vec,
    mut v28: D::F32Vec,
    mut v29: D::F32Vec,
    mut v30: D::F32Vec,
    mut v31: D::F32Vec,
    mut v32: D::F32Vec,
    mut v33: D::F32Vec,
    mut v34: D::F32Vec,
    mut v35: D::F32Vec,
    mut v36: D::F32Vec,
    mut v37: D::F32Vec,
    mut v38: D::F32Vec,
    mut v39: D::F32Vec,
    mut v40: D::F32Vec,
    mut v41: D::F32Vec,
    mut v42: D::F32Vec,
    mut v43: D::F32Vec,
    mut v44: D::F32Vec,
    mut v45: D::F32Vec,
    mut v46: D::F32Vec,
    mut v47: D::F32Vec,
    mut v48: D::F32Vec,
    mut v49: D::F32Vec,
    mut v50: D::F32Vec,
    mut v51: D::F32Vec,
    mut v52: D::F32Vec,
    mut v53: D::F32Vec,
    mut v54: D::F32Vec,
    mut v55: D::F32Vec,
    mut v56: D::F32Vec,
    mut v57: D::F32Vec,
    mut v58: D::F32Vec,
    mut v59: D::F32Vec,
    mut v60: D::F32Vec,
    mut v61: D::F32Vec,
    mut v62: D::F32Vec,
    mut v63: D::F32Vec,
    mut v64: D::F32Vec,
    mut v65: D::F32Vec,
    mut v66: D::F32Vec,
    mut v67: D::F32Vec,
    mut v68: D::F32Vec,
    mut v69: D::F32Vec,
    mut v70: D::F32Vec,
    mut v71: D::F32Vec,
    mut v72: D::F32Vec,
    mut v73: D::F32Vec,
    mut v74: D::F32Vec,
    mut v75: D::F32Vec,
    mut v76: D::F32Vec,
    mut v77: D::F32Vec,
    mut v78: D::F32Vec,
    mut v79: D::F32Vec,
    mut v80: D::F32Vec,
    mut v81: D::F32Vec,
    mut v82: D::F32Vec,
    mut v83: D::F32Vec,
    mut v84: D::F32Vec,
    mut v85: D::F32Vec,
    mut v86: D::F32Vec,
    mut v87: D::F32Vec,
    mut v88: D::F32Vec,
    mut v89: D::F32Vec,
    mut v90: D::F32Vec,
    mut v91: D::F32Vec,
    mut v92: D::F32Vec,
    mut v93: D::F32Vec,
    mut v94: D::F32Vec,
    mut v95: D::F32Vec,
    mut v96: D::F32Vec,
    mut v97: D::F32Vec,
    mut v98: D::F32Vec,
    mut v99: D::F32Vec,
    mut v100: D::F32Vec,
    mut v101: D::F32Vec,
    mut v102: D::F32Vec,
    mut v103: D::F32Vec,
    mut v104: D::F32Vec,
    mut v105: D::F32Vec,
    mut v106: D::F32Vec,
    mut v107: D::F32Vec,
    mut v108: D::F32Vec,
    mut v109: D::F32Vec,
    mut v110: D::F32Vec,
    mut v111: D::F32Vec,
    mut v112: D::F32Vec,
    mut v113: D::F32Vec,
    mut v114: D::F32Vec,
    mut v115: D::F32Vec,
    mut v116: D::F32Vec,
    mut v117: D::F32Vec,
    mut v118: D::F32Vec,
    mut v119: D::F32Vec,
    mut v120: D::F32Vec,
    mut v121: D::F32Vec,
    mut v122: D::F32Vec,
    mut v123: D::F32Vec,
    mut v124: D::F32Vec,
    mut v125: D::F32Vec,
    mut v126: D::F32Vec,
    mut v127: D::F32Vec,
) -> (
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
    D::F32Vec,
) {
    d.call(
        #[inline(always)]
        |_| {
            (
                v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v32,
                v34, v36, v38, v40, v42, v44, v46, v48, v50, v52, v54, v56, v58, v60, v62, v64,
                v66, v68, v70, v72, v74, v76, v78, v80, v82, v84, v86, v88, v90, v92, v94, v96,
                v98, v100, v102, v104, v106, v108, v110, v112, v114, v116, v118, v120, v122, v124,
                v126,
            ) = idct_64(
                d, v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v32,
                v34, v36, v38, v40, v42, v44, v46, v48, v50, v52, v54, v56, v58, v60, v62, v64,
                v66, v68, v70, v72, v74, v76, v78, v80, v82, v84, v86, v88, v90, v92, v94, v96,
                v98, v100, v102, v104, v106, v108, v110, v112, v114, v116, v118, v120, v122, v124,
                v126,
            );
        },
    );
    let mut v128 = v1 + v3;
    let mut v129 = v3 + v5;
    let mut v130 = v5 + v7;
    let mut v131 = v7 + v9;
    let mut v132 = v9 + v11;
    let mut v133 = v11 + v13;
    let mut v134 = v13 + v15;
    let mut v135 = v15 + v17;
    let mut v136 = v17 + v19;
    let mut v137 = v19 + v21;
    let mut v138 = v21 + v23;
    let mut v139 = v23 + v25;
    let mut v140 = v25 + v27;
    let mut v141 = v27 + v29;
    let mut v142 = v29 + v31;
    let mut v143 = v31 + v33;
    let mut v144 = v33 + v35;
    let mut v145 = v35 + v37;
    let mut v146 = v37 + v39;
    let mut v147 = v39 + v41;
    let mut v148 = v41 + v43;
    let mut v149 = v43 + v45;
    let mut v150 = v45 + v47;
    let mut v151 = v47 + v49;
    let mut v152 = v49 + v51;
    let mut v153 = v51 + v53;
    let mut v154 = v53 + v55;
    let mut v155 = v55 + v57;
    let mut v156 = v57 + v59;
    let mut v157 = v59 + v61;
    let mut v158 = v61 + v63;
    let mut v159 = v63 + v65;
    let mut v160 = v65 + v67;
    let mut v161 = v67 + v69;
    let mut v162 = v69 + v71;
    let mut v163 = v71 + v73;
    let mut v164 = v73 + v75;
    let mut v165 = v75 + v77;
    let mut v166 = v77 + v79;
    let mut v167 = v79 + v81;
    let mut v168 = v81 + v83;
    let mut v169 = v83 + v85;
    let mut v170 = v85 + v87;
    let mut v171 = v87 + v89;
    let mut v172 = v89 + v91;
    let mut v173 = v91 + v93;
    let mut v174 = v93 + v95;
    let mut v175 = v95 + v97;
    let mut v176 = v97 + v99;
    let mut v177 = v99 + v101;
    let mut v178 = v101 + v103;
    let mut v179 = v103 + v105;
    let mut v180 = v105 + v107;
    let mut v181 = v107 + v109;
    let mut v182 = v109 + v111;
    let mut v183 = v111 + v113;
    let mut v184 = v113 + v115;
    let mut v185 = v115 + v117;
    let mut v186 = v117 + v119;
    let mut v187 = v119 + v121;
    let mut v188 = v121 + v123;
    let mut v189 = v123 + v125;
    let mut v190 = v125 + v127;
    let mut v191 = v1 * D::F32Vec::splat(d, std::f32::consts::SQRT_2);
    d.call(
        #[inline(always)]
        |_| {
            (
                v191, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137, v138, v139, v140,
                v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152, v153, v154,
                v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166, v167, v168,
                v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181, v182,
                v183, v184, v185, v186, v187, v188, v189, v190,
            ) = idct_64(
                d, v191, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137, v138, v139,
                v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152, v153,
                v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166, v167,
                v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181,
                v182, v183, v184, v185, v186, v187, v188, v189, v190,
            );
        },
    );
    let mul = D::F32Vec::splat(d, 0.5000376519155477);
    let mut v192 = v191.mul_add(mul, v0);
    let mut v193 = v191.neg_mul_add(mul, v0);
    let mul = D::F32Vec::splat(d, 0.5003390374428216);
    let mut v194 = v128.mul_add(mul, v2);
    let mut v195 = v128.neg_mul_add(mul, v2);
    let mul = D::F32Vec::splat(d, 0.5009427176380873);
    let mut v196 = v129.mul_add(mul, v4);
    let mut v197 = v129.neg_mul_add(mul, v4);
    let mul = D::F32Vec::splat(d, 0.5018505174842379);
    let mut v198 = v130.mul_add(mul, v6);
    let mut v199 = v130.neg_mul_add(mul, v6);
    let mul = D::F32Vec::splat(d, 0.5030651913013697);
    let mut v200 = v131.mul_add(mul, v8);
    let mut v201 = v131.neg_mul_add(mul, v8);
    let mul = D::F32Vec::splat(d, 0.5045904432216454);
    let mut v202 = v132.mul_add(mul, v10);
    let mut v203 = v132.neg_mul_add(mul, v10);
    let mul = D::F32Vec::splat(d, 0.5064309549285542);
    let mut v204 = v133.mul_add(mul, v12);
    let mut v205 = v133.neg_mul_add(mul, v12);
    let mul = D::F32Vec::splat(d, 0.5085924210498143);
    let mut v206 = v134.mul_add(mul, v14);
    let mut v207 = v134.neg_mul_add(mul, v14);
    let mul = D::F32Vec::splat(d, 0.5110815927066812);
    let mut v208 = v135.mul_add(mul, v16);
    let mut v209 = v135.neg_mul_add(mul, v16);
    let mul = D::F32Vec::splat(d, 0.5139063298475396);
    let mut v210 = v136.mul_add(mul, v18);
    let mut v211 = v136.neg_mul_add(mul, v18);
    let mul = D::F32Vec::splat(d, 0.5170756631334912);
    let mut v212 = v137.mul_add(mul, v20);
    let mut v213 = v137.neg_mul_add(mul, v20);
    let mul = D::F32Vec::splat(d, 0.5205998663018917);
    let mut v214 = v138.mul_add(mul, v22);
    let mut v215 = v138.neg_mul_add(mul, v22);
    let mul = D::F32Vec::splat(d, 0.5244905401147240);
    let mut v216 = v139.mul_add(mul, v24);
    let mut v217 = v139.neg_mul_add(mul, v24);
    let mul = D::F32Vec::splat(d, 0.5287607092074876);
    let mut v218 = v140.mul_add(mul, v26);
    let mut v219 = v140.neg_mul_add(mul, v26);
    let mul = D::F32Vec::splat(d, 0.5334249333971333);
    let mut v220 = v141.mul_add(mul, v28);
    let mut v221 = v141.neg_mul_add(mul, v28);
    let mul = D::F32Vec::splat(d, 0.5384994352919840);
    let mut v222 = v142.mul_add(mul, v30);
    let mut v223 = v142.neg_mul_add(mul, v30);
    let mul = D::F32Vec::splat(d, 0.5440022463817783);
    let mut v224 = v143.mul_add(mul, v32);
    let mut v225 = v143.neg_mul_add(mul, v32);
    let mul = D::F32Vec::splat(d, 0.5499533741832360);
    let mut v226 = v144.mul_add(mul, v34);
    let mut v227 = v144.neg_mul_add(mul, v34);
    let mul = D::F32Vec::splat(d, 0.5563749934898856);
    let mut v228 = v145.mul_add(mul, v36);
    let mut v229 = v145.neg_mul_add(mul, v36);
    let mul = D::F32Vec::splat(d, 0.5632916653417023);
    let mut v230 = v146.mul_add(mul, v38);
    let mut v231 = v146.neg_mul_add(mul, v38);
    let mul = D::F32Vec::splat(d, 0.5707305880121454);
    let mut v232 = v147.mul_add(mul, v40);
    let mut v233 = v147.neg_mul_add(mul, v40);
    let mul = D::F32Vec::splat(d, 0.5787218851348208);
    let mut v234 = v148.mul_add(mul, v42);
    let mut v235 = v148.neg_mul_add(mul, v42);
    let mul = D::F32Vec::splat(d, 0.5872989370937893);
    let mut v236 = v149.mul_add(mul, v44);
    let mut v237 = v149.neg_mul_add(mul, v44);
    let mul = D::F32Vec::splat(d, 0.5964987630244563);
    let mut v238 = v150.mul_add(mul, v46);
    let mut v239 = v150.neg_mul_add(mul, v46);
    let mul = D::F32Vec::splat(d, 0.6063624622721460);
    let mut v240 = v151.mul_add(mul, v48);
    let mut v241 = v151.neg_mul_add(mul, v48);
    let mul = D::F32Vec::splat(d, 0.6169357260050706);
    let mut v242 = v152.mul_add(mul, v50);
    let mut v243 = v152.neg_mul_add(mul, v50);
    let mul = D::F32Vec::splat(d, 0.6282694319707711);
    let mut v244 = v153.mul_add(mul, v52);
    let mut v245 = v153.neg_mul_add(mul, v52);
    let mul = D::F32Vec::splat(d, 0.6404203382416639);
    let mut v246 = v154.mul_add(mul, v54);
    let mut v247 = v154.neg_mul_add(mul, v54);
    let mul = D::F32Vec::splat(d, 0.6534518953751283);
    let mut v248 = v155.mul_add(mul, v56);
    let mut v249 = v155.neg_mul_add(mul, v56);
    let mul = D::F32Vec::splat(d, 0.6674352009263413);
    let mut v250 = v156.mul_add(mul, v58);
    let mut v251 = v156.neg_mul_add(mul, v58);
    let mul = D::F32Vec::splat(d, 0.6824501259764195);
    let mut v252 = v157.mul_add(mul, v60);
    let mut v253 = v157.neg_mul_add(mul, v60);
    let mul = D::F32Vec::splat(d, 0.6985866506472291);
    let mut v254 = v158.mul_add(mul, v62);
    let mut v255 = v158.neg_mul_add(mul, v62);
    let mul = D::F32Vec::splat(d, 0.7159464549705746);
    let mut v256 = v159.mul_add(mul, v64);
    let mut v257 = v159.neg_mul_add(mul, v64);
    let mul = D::F32Vec::splat(d, 0.7346448236478627);
    let mut v258 = v160.mul_add(mul, v66);
    let mut v259 = v160.neg_mul_add(mul, v66);
    let mul = D::F32Vec::splat(d, 0.7548129391165311);
    let mut v260 = v161.mul_add(mul, v68);
    let mut v261 = v161.neg_mul_add(mul, v68);
    let mul = D::F32Vec::splat(d, 0.7766006582339630);
    let mut v262 = v162.mul_add(mul, v70);
    let mut v263 = v162.neg_mul_add(mul, v70);
    let mul = D::F32Vec::splat(d, 0.8001798956216941);
    let mut v264 = v163.mul_add(mul, v72);
    let mut v265 = v163.neg_mul_add(mul, v72);
    let mul = D::F32Vec::splat(d, 0.8257487738627852);
    let mut v266 = v164.mul_add(mul, v74);
    let mut v267 = v164.neg_mul_add(mul, v74);
    let mul = D::F32Vec::splat(d, 0.8535367510066064);
    let mut v268 = v165.mul_add(mul, v76);
    let mut v269 = v165.neg_mul_add(mul, v76);
    let mul = D::F32Vec::splat(d, 0.8838110045596234);
    let mut v270 = v166.mul_add(mul, v78);
    let mut v271 = v166.neg_mul_add(mul, v78);
    let mul = D::F32Vec::splat(d, 0.9168844461846523);
    let mut v272 = v167.mul_add(mul, v80);
    let mut v273 = v167.neg_mul_add(mul, v80);
    let mul = D::F32Vec::splat(d, 0.9531258743921193);
    let mut v274 = v168.mul_add(mul, v82);
    let mut v275 = v168.neg_mul_add(mul, v82);
    let mul = D::F32Vec::splat(d, 0.9929729612675466);
    let mut v276 = v169.mul_add(mul, v84);
    let mut v277 = v169.neg_mul_add(mul, v84);
    let mul = D::F32Vec::splat(d, 1.0369490409103890);
    let mut v278 = v170.mul_add(mul, v86);
    let mut v279 = v170.neg_mul_add(mul, v86);
    let mul = D::F32Vec::splat(d, 1.0856850642580145);
    let mut v280 = v171.mul_add(mul, v88);
    let mut v281 = v171.neg_mul_add(mul, v88);
    let mul = D::F32Vec::splat(d, 1.1399486751015042);
    let mut v282 = v172.mul_add(mul, v90);
    let mut v283 = v172.neg_mul_add(mul, v90);
    let mul = D::F32Vec::splat(d, 1.2006832557294167);
    let mut v284 = v173.mul_add(mul, v92);
    let mut v285 = v173.neg_mul_add(mul, v92);
    let mul = D::F32Vec::splat(d, 1.2690611716991191);
    let mut v286 = v174.mul_add(mul, v94);
    let mut v287 = v174.neg_mul_add(mul, v94);
    let mul = D::F32Vec::splat(d, 1.3465576282062861);
    let mut v288 = v175.mul_add(mul, v96);
    let mut v289 = v175.neg_mul_add(mul, v96);
    let mul = D::F32Vec::splat(d, 1.4350550884414341);
    let mut v290 = v176.mul_add(mul, v98);
    let mut v291 = v176.neg_mul_add(mul, v98);
    let mul = D::F32Vec::splat(d, 1.5369941008524954);
    let mut v292 = v177.mul_add(mul, v100);
    let mut v293 = v177.neg_mul_add(mul, v100);
    let mul = D::F32Vec::splat(d, 1.6555965242641195);
    let mut v294 = v178.mul_add(mul, v102);
    let mut v295 = v178.neg_mul_add(mul, v102);
    let mul = D::F32Vec::splat(d, 1.7952052190778898);
    let mut v296 = v179.mul_add(mul, v104);
    let mut v297 = v179.neg_mul_add(mul, v104);
    let mul = D::F32Vec::splat(d, 1.9618178485711659);
    let mut v298 = v180.mul_add(mul, v106);
    let mut v299 = v180.neg_mul_add(mul, v106);
    let mul = D::F32Vec::splat(d, 2.1639578187519790);
    let mut v300 = v181.mul_add(mul, v108);
    let mut v301 = v181.neg_mul_add(mul, v108);
    let mul = D::F32Vec::splat(d, 2.4141600002500763);
    let mut v302 = v182.mul_add(mul, v110);
    let mut v303 = v182.neg_mul_add(mul, v110);
    let mul = D::F32Vec::splat(d, 2.7316450287739396);
    let mut v304 = v183.mul_add(mul, v112);
    let mut v305 = v183.neg_mul_add(mul, v112);
    let mul = D::F32Vec::splat(d, 3.1474621917819090);
    let mut v306 = v184.mul_add(mul, v114);
    let mut v307 = v184.neg_mul_add(mul, v114);
    let mul = D::F32Vec::splat(d, 3.7152427383269746);
    let mut v308 = v185.mul_add(mul, v116);
    let mut v309 = v185.neg_mul_add(mul, v116);
    let mul = D::F32Vec::splat(d, 4.5362909369693565);
    let mut v310 = v186.mul_add(mul, v118);
    let mut v311 = v186.neg_mul_add(mul, v118);
    let mul = D::F32Vec::splat(d, 5.8276883778446544);
    let mut v312 = v187.mul_add(mul, v120);
    let mut v313 = v187.neg_mul_add(mul, v120);
    let mul = D::F32Vec::splat(d, 8.1538486024668142);
    let mut v314 = v188.mul_add(mul, v122);
    let mut v315 = v188.neg_mul_add(mul, v122);
    let mul = D::F32Vec::splat(d, 13.5842902572844597);
    let mut v316 = v189.mul_add(mul, v124);
    let mut v317 = v189.neg_mul_add(mul, v124);
    let mul = D::F32Vec::splat(d, 40.7446881033518338);
    let mut v318 = v190.mul_add(mul, v126);
    let mut v319 = v190.neg_mul_add(mul, v126);
    (
        v192, v194, v196, v198, v200, v202, v204, v206, v208, v210, v212, v214, v216, v218, v220,
        v222, v224, v226, v228, v230, v232, v234, v236, v238, v240, v242, v244, v246, v248, v250,
        v252, v254, v256, v258, v260, v262, v264, v266, v268, v270, v272, v274, v276, v278, v280,
        v282, v284, v286, v288, v290, v292, v294, v296, v298, v300, v302, v304, v306, v308, v310,
        v312, v314, v316, v318, v319, v317, v315, v313, v311, v309, v307, v305, v303, v301, v299,
        v297, v295, v293, v291, v289, v287, v285, v283, v281, v279, v277, v275, v273, v271, v269,
        v267, v265, v263, v261, v259, v257, v255, v253, v251, v249, v247, v245, v243, v241, v239,
        v237, v235, v233, v231, v229, v227, v225, v223, v221, v219, v217, v215, v213, v211, v209,
        v207, v205, v203, v201, v199, v197, v195, v193,
    )
}

#[inline(always)]
pub(super) fn do_idct_128<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
    stride: usize,
) {
    assert!(data.len() > 127 * stride);
    let mut v0 = D::F32Vec::load_array(d, &data[0 * stride]);
    let mut v1 = D::F32Vec::load_array(d, &data[1 * stride]);
    let mut v2 = D::F32Vec::load_array(d, &data[2 * stride]);
    let mut v3 = D::F32Vec::load_array(d, &data[3 * stride]);
    let mut v4 = D::F32Vec::load_array(d, &data[4 * stride]);
    let mut v5 = D::F32Vec::load_array(d, &data[5 * stride]);
    let mut v6 = D::F32Vec::load_array(d, &data[6 * stride]);
    let mut v7 = D::F32Vec::load_array(d, &data[7 * stride]);
    let mut v8 = D::F32Vec::load_array(d, &data[8 * stride]);
    let mut v9 = D::F32Vec::load_array(d, &data[9 * stride]);
    let mut v10 = D::F32Vec::load_array(d, &data[10 * stride]);
    let mut v11 = D::F32Vec::load_array(d, &data[11 * stride]);
    let mut v12 = D::F32Vec::load_array(d, &data[12 * stride]);
    let mut v13 = D::F32Vec::load_array(d, &data[13 * stride]);
    let mut v14 = D::F32Vec::load_array(d, &data[14 * stride]);
    let mut v15 = D::F32Vec::load_array(d, &data[15 * stride]);
    let mut v16 = D::F32Vec::load_array(d, &data[16 * stride]);
    let mut v17 = D::F32Vec::load_array(d, &data[17 * stride]);
    let mut v18 = D::F32Vec::load_array(d, &data[18 * stride]);
    let mut v19 = D::F32Vec::load_array(d, &data[19 * stride]);
    let mut v20 = D::F32Vec::load_array(d, &data[20 * stride]);
    let mut v21 = D::F32Vec::load_array(d, &data[21 * stride]);
    let mut v22 = D::F32Vec::load_array(d, &data[22 * stride]);
    let mut v23 = D::F32Vec::load_array(d, &data[23 * stride]);
    let mut v24 = D::F32Vec::load_array(d, &data[24 * stride]);
    let mut v25 = D::F32Vec::load_array(d, &data[25 * stride]);
    let mut v26 = D::F32Vec::load_array(d, &data[26 * stride]);
    let mut v27 = D::F32Vec::load_array(d, &data[27 * stride]);
    let mut v28 = D::F32Vec::load_array(d, &data[28 * stride]);
    let mut v29 = D::F32Vec::load_array(d, &data[29 * stride]);
    let mut v30 = D::F32Vec::load_array(d, &data[30 * stride]);
    let mut v31 = D::F32Vec::load_array(d, &data[31 * stride]);
    let mut v32 = D::F32Vec::load_array(d, &data[32 * stride]);
    let mut v33 = D::F32Vec::load_array(d, &data[33 * stride]);
    let mut v34 = D::F32Vec::load_array(d, &data[34 * stride]);
    let mut v35 = D::F32Vec::load_array(d, &data[35 * stride]);
    let mut v36 = D::F32Vec::load_array(d, &data[36 * stride]);
    let mut v37 = D::F32Vec::load_array(d, &data[37 * stride]);
    let mut v38 = D::F32Vec::load_array(d, &data[38 * stride]);
    let mut v39 = D::F32Vec::load_array(d, &data[39 * stride]);
    let mut v40 = D::F32Vec::load_array(d, &data[40 * stride]);
    let mut v41 = D::F32Vec::load_array(d, &data[41 * stride]);
    let mut v42 = D::F32Vec::load_array(d, &data[42 * stride]);
    let mut v43 = D::F32Vec::load_array(d, &data[43 * stride]);
    let mut v44 = D::F32Vec::load_array(d, &data[44 * stride]);
    let mut v45 = D::F32Vec::load_array(d, &data[45 * stride]);
    let mut v46 = D::F32Vec::load_array(d, &data[46 * stride]);
    let mut v47 = D::F32Vec::load_array(d, &data[47 * stride]);
    let mut v48 = D::F32Vec::load_array(d, &data[48 * stride]);
    let mut v49 = D::F32Vec::load_array(d, &data[49 * stride]);
    let mut v50 = D::F32Vec::load_array(d, &data[50 * stride]);
    let mut v51 = D::F32Vec::load_array(d, &data[51 * stride]);
    let mut v52 = D::F32Vec::load_array(d, &data[52 * stride]);
    let mut v53 = D::F32Vec::load_array(d, &data[53 * stride]);
    let mut v54 = D::F32Vec::load_array(d, &data[54 * stride]);
    let mut v55 = D::F32Vec::load_array(d, &data[55 * stride]);
    let mut v56 = D::F32Vec::load_array(d, &data[56 * stride]);
    let mut v57 = D::F32Vec::load_array(d, &data[57 * stride]);
    let mut v58 = D::F32Vec::load_array(d, &data[58 * stride]);
    let mut v59 = D::F32Vec::load_array(d, &data[59 * stride]);
    let mut v60 = D::F32Vec::load_array(d, &data[60 * stride]);
    let mut v61 = D::F32Vec::load_array(d, &data[61 * stride]);
    let mut v62 = D::F32Vec::load_array(d, &data[62 * stride]);
    let mut v63 = D::F32Vec::load_array(d, &data[63 * stride]);
    let mut v64 = D::F32Vec::load_array(d, &data[64 * stride]);
    let mut v65 = D::F32Vec::load_array(d, &data[65 * stride]);
    let mut v66 = D::F32Vec::load_array(d, &data[66 * stride]);
    let mut v67 = D::F32Vec::load_array(d, &data[67 * stride]);
    let mut v68 = D::F32Vec::load_array(d, &data[68 * stride]);
    let mut v69 = D::F32Vec::load_array(d, &data[69 * stride]);
    let mut v70 = D::F32Vec::load_array(d, &data[70 * stride]);
    let mut v71 = D::F32Vec::load_array(d, &data[71 * stride]);
    let mut v72 = D::F32Vec::load_array(d, &data[72 * stride]);
    let mut v73 = D::F32Vec::load_array(d, &data[73 * stride]);
    let mut v74 = D::F32Vec::load_array(d, &data[74 * stride]);
    let mut v75 = D::F32Vec::load_array(d, &data[75 * stride]);
    let mut v76 = D::F32Vec::load_array(d, &data[76 * stride]);
    let mut v77 = D::F32Vec::load_array(d, &data[77 * stride]);
    let mut v78 = D::F32Vec::load_array(d, &data[78 * stride]);
    let mut v79 = D::F32Vec::load_array(d, &data[79 * stride]);
    let mut v80 = D::F32Vec::load_array(d, &data[80 * stride]);
    let mut v81 = D::F32Vec::load_array(d, &data[81 * stride]);
    let mut v82 = D::F32Vec::load_array(d, &data[82 * stride]);
    let mut v83 = D::F32Vec::load_array(d, &data[83 * stride]);
    let mut v84 = D::F32Vec::load_array(d, &data[84 * stride]);
    let mut v85 = D::F32Vec::load_array(d, &data[85 * stride]);
    let mut v86 = D::F32Vec::load_array(d, &data[86 * stride]);
    let mut v87 = D::F32Vec::load_array(d, &data[87 * stride]);
    let mut v88 = D::F32Vec::load_array(d, &data[88 * stride]);
    let mut v89 = D::F32Vec::load_array(d, &data[89 * stride]);
    let mut v90 = D::F32Vec::load_array(d, &data[90 * stride]);
    let mut v91 = D::F32Vec::load_array(d, &data[91 * stride]);
    let mut v92 = D::F32Vec::load_array(d, &data[92 * stride]);
    let mut v93 = D::F32Vec::load_array(d, &data[93 * stride]);
    let mut v94 = D::F32Vec::load_array(d, &data[94 * stride]);
    let mut v95 = D::F32Vec::load_array(d, &data[95 * stride]);
    let mut v96 = D::F32Vec::load_array(d, &data[96 * stride]);
    let mut v97 = D::F32Vec::load_array(d, &data[97 * stride]);
    let mut v98 = D::F32Vec::load_array(d, &data[98 * stride]);
    let mut v99 = D::F32Vec::load_array(d, &data[99 * stride]);
    let mut v100 = D::F32Vec::load_array(d, &data[100 * stride]);
    let mut v101 = D::F32Vec::load_array(d, &data[101 * stride]);
    let mut v102 = D::F32Vec::load_array(d, &data[102 * stride]);
    let mut v103 = D::F32Vec::load_array(d, &data[103 * stride]);
    let mut v104 = D::F32Vec::load_array(d, &data[104 * stride]);
    let mut v105 = D::F32Vec::load_array(d, &data[105 * stride]);
    let mut v106 = D::F32Vec::load_array(d, &data[106 * stride]);
    let mut v107 = D::F32Vec::load_array(d, &data[107 * stride]);
    let mut v108 = D::F32Vec::load_array(d, &data[108 * stride]);
    let mut v109 = D::F32Vec::load_array(d, &data[109 * stride]);
    let mut v110 = D::F32Vec::load_array(d, &data[110 * stride]);
    let mut v111 = D::F32Vec::load_array(d, &data[111 * stride]);
    let mut v112 = D::F32Vec::load_array(d, &data[112 * stride]);
    let mut v113 = D::F32Vec::load_array(d, &data[113 * stride]);
    let mut v114 = D::F32Vec::load_array(d, &data[114 * stride]);
    let mut v115 = D::F32Vec::load_array(d, &data[115 * stride]);
    let mut v116 = D::F32Vec::load_array(d, &data[116 * stride]);
    let mut v117 = D::F32Vec::load_array(d, &data[117 * stride]);
    let mut v118 = D::F32Vec::load_array(d, &data[118 * stride]);
    let mut v119 = D::F32Vec::load_array(d, &data[119 * stride]);
    let mut v120 = D::F32Vec::load_array(d, &data[120 * stride]);
    let mut v121 = D::F32Vec::load_array(d, &data[121 * stride]);
    let mut v122 = D::F32Vec::load_array(d, &data[122 * stride]);
    let mut v123 = D::F32Vec::load_array(d, &data[123 * stride]);
    let mut v124 = D::F32Vec::load_array(d, &data[124 * stride]);
    let mut v125 = D::F32Vec::load_array(d, &data[125 * stride]);
    let mut v126 = D::F32Vec::load_array(d, &data[126 * stride]);
    let mut v127 = D::F32Vec::load_array(d, &data[127 * stride]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73,
        v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91,
        v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107,
        v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122,
        v123, v124, v125, v126, v127,
    ) = idct_128(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72,
        v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90,
        v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106,
        v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121,
        v122, v123, v124, v125, v126, v127,
    );
    v0.store_array(&mut data[0 * stride]);
    v1.store_array(&mut data[1 * stride]);
    v2.store_array(&mut data[2 * stride]);
    v3.store_array(&mut data[3 * stride]);
    v4.store_array(&mut data[4 * stride]);
    v5.store_array(&mut data[5 * stride]);
    v6.store_array(&mut data[6 * stride]);
    v7.store_array(&mut data[7 * stride]);
    v8.store_array(&mut data[8 * stride]);
    v9.store_array(&mut data[9 * stride]);
    v10.store_array(&mut data[10 * stride]);
    v11.store_array(&mut data[11 * stride]);
    v12.store_array(&mut data[12 * stride]);
    v13.store_array(&mut data[13 * stride]);
    v14.store_array(&mut data[14 * stride]);
    v15.store_array(&mut data[15 * stride]);
    v16.store_array(&mut data[16 * stride]);
    v17.store_array(&mut data[17 * stride]);
    v18.store_array(&mut data[18 * stride]);
    v19.store_array(&mut data[19 * stride]);
    v20.store_array(&mut data[20 * stride]);
    v21.store_array(&mut data[21 * stride]);
    v22.store_array(&mut data[22 * stride]);
    v23.store_array(&mut data[23 * stride]);
    v24.store_array(&mut data[24 * stride]);
    v25.store_array(&mut data[25 * stride]);
    v26.store_array(&mut data[26 * stride]);
    v27.store_array(&mut data[27 * stride]);
    v28.store_array(&mut data[28 * stride]);
    v29.store_array(&mut data[29 * stride]);
    v30.store_array(&mut data[30 * stride]);
    v31.store_array(&mut data[31 * stride]);
    v32.store_array(&mut data[32 * stride]);
    v33.store_array(&mut data[33 * stride]);
    v34.store_array(&mut data[34 * stride]);
    v35.store_array(&mut data[35 * stride]);
    v36.store_array(&mut data[36 * stride]);
    v37.store_array(&mut data[37 * stride]);
    v38.store_array(&mut data[38 * stride]);
    v39.store_array(&mut data[39 * stride]);
    v40.store_array(&mut data[40 * stride]);
    v41.store_array(&mut data[41 * stride]);
    v42.store_array(&mut data[42 * stride]);
    v43.store_array(&mut data[43 * stride]);
    v44.store_array(&mut data[44 * stride]);
    v45.store_array(&mut data[45 * stride]);
    v46.store_array(&mut data[46 * stride]);
    v47.store_array(&mut data[47 * stride]);
    v48.store_array(&mut data[48 * stride]);
    v49.store_array(&mut data[49 * stride]);
    v50.store_array(&mut data[50 * stride]);
    v51.store_array(&mut data[51 * stride]);
    v52.store_array(&mut data[52 * stride]);
    v53.store_array(&mut data[53 * stride]);
    v54.store_array(&mut data[54 * stride]);
    v55.store_array(&mut data[55 * stride]);
    v56.store_array(&mut data[56 * stride]);
    v57.store_array(&mut data[57 * stride]);
    v58.store_array(&mut data[58 * stride]);
    v59.store_array(&mut data[59 * stride]);
    v60.store_array(&mut data[60 * stride]);
    v61.store_array(&mut data[61 * stride]);
    v62.store_array(&mut data[62 * stride]);
    v63.store_array(&mut data[63 * stride]);
    v64.store_array(&mut data[64 * stride]);
    v65.store_array(&mut data[65 * stride]);
    v66.store_array(&mut data[66 * stride]);
    v67.store_array(&mut data[67 * stride]);
    v68.store_array(&mut data[68 * stride]);
    v69.store_array(&mut data[69 * stride]);
    v70.store_array(&mut data[70 * stride]);
    v71.store_array(&mut data[71 * stride]);
    v72.store_array(&mut data[72 * stride]);
    v73.store_array(&mut data[73 * stride]);
    v74.store_array(&mut data[74 * stride]);
    v75.store_array(&mut data[75 * stride]);
    v76.store_array(&mut data[76 * stride]);
    v77.store_array(&mut data[77 * stride]);
    v78.store_array(&mut data[78 * stride]);
    v79.store_array(&mut data[79 * stride]);
    v80.store_array(&mut data[80 * stride]);
    v81.store_array(&mut data[81 * stride]);
    v82.store_array(&mut data[82 * stride]);
    v83.store_array(&mut data[83 * stride]);
    v84.store_array(&mut data[84 * stride]);
    v85.store_array(&mut data[85 * stride]);
    v86.store_array(&mut data[86 * stride]);
    v87.store_array(&mut data[87 * stride]);
    v88.store_array(&mut data[88 * stride]);
    v89.store_array(&mut data[89 * stride]);
    v90.store_array(&mut data[90 * stride]);
    v91.store_array(&mut data[91 * stride]);
    v92.store_array(&mut data[92 * stride]);
    v93.store_array(&mut data[93 * stride]);
    v94.store_array(&mut data[94 * stride]);
    v95.store_array(&mut data[95 * stride]);
    v96.store_array(&mut data[96 * stride]);
    v97.store_array(&mut data[97 * stride]);
    v98.store_array(&mut data[98 * stride]);
    v99.store_array(&mut data[99 * stride]);
    v100.store_array(&mut data[100 * stride]);
    v101.store_array(&mut data[101 * stride]);
    v102.store_array(&mut data[102 * stride]);
    v103.store_array(&mut data[103 * stride]);
    v104.store_array(&mut data[104 * stride]);
    v105.store_array(&mut data[105 * stride]);
    v106.store_array(&mut data[106 * stride]);
    v107.store_array(&mut data[107 * stride]);
    v108.store_array(&mut data[108 * stride]);
    v109.store_array(&mut data[109 * stride]);
    v110.store_array(&mut data[110 * stride]);
    v111.store_array(&mut data[111 * stride]);
    v112.store_array(&mut data[112 * stride]);
    v113.store_array(&mut data[113 * stride]);
    v114.store_array(&mut data[114 * stride]);
    v115.store_array(&mut data[115 * stride]);
    v116.store_array(&mut data[116 * stride]);
    v117.store_array(&mut data[117 * stride]);
    v118.store_array(&mut data[118 * stride]);
    v119.store_array(&mut data[119 * stride]);
    v120.store_array(&mut data[120 * stride]);
    v121.store_array(&mut data[121 * stride]);
    v122.store_array(&mut data[122 * stride]);
    v123.store_array(&mut data[123 * stride]);
    v124.store_array(&mut data[124 * stride]);
    v125.store_array(&mut data[125 * stride]);
    v126.store_array(&mut data[126 * stride]);
    v127.store_array(&mut data[127 * stride]);
}

#[inline(always)]
pub(super) fn do_idct_128_rowblock<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
) {
    assert!(data.len() >= 128);
    const { assert!(128usize.is_multiple_of(D::F32Vec::LEN)) };
    let row_stride = 128 / D::F32Vec::LEN;
    let mut v0 = D::F32Vec::load_array(
        d,
        &data[row_stride * (0 % D::F32Vec::LEN) + (0 / D::F32Vec::LEN)],
    );
    let mut v1 = D::F32Vec::load_array(
        d,
        &data[row_stride * (1 % D::F32Vec::LEN) + (1 / D::F32Vec::LEN)],
    );
    let mut v2 = D::F32Vec::load_array(
        d,
        &data[row_stride * (2 % D::F32Vec::LEN) + (2 / D::F32Vec::LEN)],
    );
    let mut v3 = D::F32Vec::load_array(
        d,
        &data[row_stride * (3 % D::F32Vec::LEN) + (3 / D::F32Vec::LEN)],
    );
    let mut v4 = D::F32Vec::load_array(
        d,
        &data[row_stride * (4 % D::F32Vec::LEN) + (4 / D::F32Vec::LEN)],
    );
    let mut v5 = D::F32Vec::load_array(
        d,
        &data[row_stride * (5 % D::F32Vec::LEN) + (5 / D::F32Vec::LEN)],
    );
    let mut v6 = D::F32Vec::load_array(
        d,
        &data[row_stride * (6 % D::F32Vec::LEN) + (6 / D::F32Vec::LEN)],
    );
    let mut v7 = D::F32Vec::load_array(
        d,
        &data[row_stride * (7 % D::F32Vec::LEN) + (7 / D::F32Vec::LEN)],
    );
    let mut v8 = D::F32Vec::load_array(
        d,
        &data[row_stride * (8 % D::F32Vec::LEN) + (8 / D::F32Vec::LEN)],
    );
    let mut v9 = D::F32Vec::load_array(
        d,
        &data[row_stride * (9 % D::F32Vec::LEN) + (9 / D::F32Vec::LEN)],
    );
    let mut v10 = D::F32Vec::load_array(
        d,
        &data[row_stride * (10 % D::F32Vec::LEN) + (10 / D::F32Vec::LEN)],
    );
    let mut v11 = D::F32Vec::load_array(
        d,
        &data[row_stride * (11 % D::F32Vec::LEN) + (11 / D::F32Vec::LEN)],
    );
    let mut v12 = D::F32Vec::load_array(
        d,
        &data[row_stride * (12 % D::F32Vec::LEN) + (12 / D::F32Vec::LEN)],
    );
    let mut v13 = D::F32Vec::load_array(
        d,
        &data[row_stride * (13 % D::F32Vec::LEN) + (13 / D::F32Vec::LEN)],
    );
    let mut v14 = D::F32Vec::load_array(
        d,
        &data[row_stride * (14 % D::F32Vec::LEN) + (14 / D::F32Vec::LEN)],
    );
    let mut v15 = D::F32Vec::load_array(
        d,
        &data[row_stride * (15 % D::F32Vec::LEN) + (15 / D::F32Vec::LEN)],
    );
    let mut v16 = D::F32Vec::load_array(
        d,
        &data[row_stride * (16 % D::F32Vec::LEN) + (16 / D::F32Vec::LEN)],
    );
    let mut v17 = D::F32Vec::load_array(
        d,
        &data[row_stride * (17 % D::F32Vec::LEN) + (17 / D::F32Vec::LEN)],
    );
    let mut v18 = D::F32Vec::load_array(
        d,
        &data[row_stride * (18 % D::F32Vec::LEN) + (18 / D::F32Vec::LEN)],
    );
    let mut v19 = D::F32Vec::load_array(
        d,
        &data[row_stride * (19 % D::F32Vec::LEN) + (19 / D::F32Vec::LEN)],
    );
    let mut v20 = D::F32Vec::load_array(
        d,
        &data[row_stride * (20 % D::F32Vec::LEN) + (20 / D::F32Vec::LEN)],
    );
    let mut v21 = D::F32Vec::load_array(
        d,
        &data[row_stride * (21 % D::F32Vec::LEN) + (21 / D::F32Vec::LEN)],
    );
    let mut v22 = D::F32Vec::load_array(
        d,
        &data[row_stride * (22 % D::F32Vec::LEN) + (22 / D::F32Vec::LEN)],
    );
    let mut v23 = D::F32Vec::load_array(
        d,
        &data[row_stride * (23 % D::F32Vec::LEN) + (23 / D::F32Vec::LEN)],
    );
    let mut v24 = D::F32Vec::load_array(
        d,
        &data[row_stride * (24 % D::F32Vec::LEN) + (24 / D::F32Vec::LEN)],
    );
    let mut v25 = D::F32Vec::load_array(
        d,
        &data[row_stride * (25 % D::F32Vec::LEN) + (25 / D::F32Vec::LEN)],
    );
    let mut v26 = D::F32Vec::load_array(
        d,
        &data[row_stride * (26 % D::F32Vec::LEN) + (26 / D::F32Vec::LEN)],
    );
    let mut v27 = D::F32Vec::load_array(
        d,
        &data[row_stride * (27 % D::F32Vec::LEN) + (27 / D::F32Vec::LEN)],
    );
    let mut v28 = D::F32Vec::load_array(
        d,
        &data[row_stride * (28 % D::F32Vec::LEN) + (28 / D::F32Vec::LEN)],
    );
    let mut v29 = D::F32Vec::load_array(
        d,
        &data[row_stride * (29 % D::F32Vec::LEN) + (29 / D::F32Vec::LEN)],
    );
    let mut v30 = D::F32Vec::load_array(
        d,
        &data[row_stride * (30 % D::F32Vec::LEN) + (30 / D::F32Vec::LEN)],
    );
    let mut v31 = D::F32Vec::load_array(
        d,
        &data[row_stride * (31 % D::F32Vec::LEN) + (31 / D::F32Vec::LEN)],
    );
    let mut v32 = D::F32Vec::load_array(
        d,
        &data[row_stride * (32 % D::F32Vec::LEN) + (32 / D::F32Vec::LEN)],
    );
    let mut v33 = D::F32Vec::load_array(
        d,
        &data[row_stride * (33 % D::F32Vec::LEN) + (33 / D::F32Vec::LEN)],
    );
    let mut v34 = D::F32Vec::load_array(
        d,
        &data[row_stride * (34 % D::F32Vec::LEN) + (34 / D::F32Vec::LEN)],
    );
    let mut v35 = D::F32Vec::load_array(
        d,
        &data[row_stride * (35 % D::F32Vec::LEN) + (35 / D::F32Vec::LEN)],
    );
    let mut v36 = D::F32Vec::load_array(
        d,
        &data[row_stride * (36 % D::F32Vec::LEN) + (36 / D::F32Vec::LEN)],
    );
    let mut v37 = D::F32Vec::load_array(
        d,
        &data[row_stride * (37 % D::F32Vec::LEN) + (37 / D::F32Vec::LEN)],
    );
    let mut v38 = D::F32Vec::load_array(
        d,
        &data[row_stride * (38 % D::F32Vec::LEN) + (38 / D::F32Vec::LEN)],
    );
    let mut v39 = D::F32Vec::load_array(
        d,
        &data[row_stride * (39 % D::F32Vec::LEN) + (39 / D::F32Vec::LEN)],
    );
    let mut v40 = D::F32Vec::load_array(
        d,
        &data[row_stride * (40 % D::F32Vec::LEN) + (40 / D::F32Vec::LEN)],
    );
    let mut v41 = D::F32Vec::load_array(
        d,
        &data[row_stride * (41 % D::F32Vec::LEN) + (41 / D::F32Vec::LEN)],
    );
    let mut v42 = D::F32Vec::load_array(
        d,
        &data[row_stride * (42 % D::F32Vec::LEN) + (42 / D::F32Vec::LEN)],
    );
    let mut v43 = D::F32Vec::load_array(
        d,
        &data[row_stride * (43 % D::F32Vec::LEN) + (43 / D::F32Vec::LEN)],
    );
    let mut v44 = D::F32Vec::load_array(
        d,
        &data[row_stride * (44 % D::F32Vec::LEN) + (44 / D::F32Vec::LEN)],
    );
    let mut v45 = D::F32Vec::load_array(
        d,
        &data[row_stride * (45 % D::F32Vec::LEN) + (45 / D::F32Vec::LEN)],
    );
    let mut v46 = D::F32Vec::load_array(
        d,
        &data[row_stride * (46 % D::F32Vec::LEN) + (46 / D::F32Vec::LEN)],
    );
    let mut v47 = D::F32Vec::load_array(
        d,
        &data[row_stride * (47 % D::F32Vec::LEN) + (47 / D::F32Vec::LEN)],
    );
    let mut v48 = D::F32Vec::load_array(
        d,
        &data[row_stride * (48 % D::F32Vec::LEN) + (48 / D::F32Vec::LEN)],
    );
    let mut v49 = D::F32Vec::load_array(
        d,
        &data[row_stride * (49 % D::F32Vec::LEN) + (49 / D::F32Vec::LEN)],
    );
    let mut v50 = D::F32Vec::load_array(
        d,
        &data[row_stride * (50 % D::F32Vec::LEN) + (50 / D::F32Vec::LEN)],
    );
    let mut v51 = D::F32Vec::load_array(
        d,
        &data[row_stride * (51 % D::F32Vec::LEN) + (51 / D::F32Vec::LEN)],
    );
    let mut v52 = D::F32Vec::load_array(
        d,
        &data[row_stride * (52 % D::F32Vec::LEN) + (52 / D::F32Vec::LEN)],
    );
    let mut v53 = D::F32Vec::load_array(
        d,
        &data[row_stride * (53 % D::F32Vec::LEN) + (53 / D::F32Vec::LEN)],
    );
    let mut v54 = D::F32Vec::load_array(
        d,
        &data[row_stride * (54 % D::F32Vec::LEN) + (54 / D::F32Vec::LEN)],
    );
    let mut v55 = D::F32Vec::load_array(
        d,
        &data[row_stride * (55 % D::F32Vec::LEN) + (55 / D::F32Vec::LEN)],
    );
    let mut v56 = D::F32Vec::load_array(
        d,
        &data[row_stride * (56 % D::F32Vec::LEN) + (56 / D::F32Vec::LEN)],
    );
    let mut v57 = D::F32Vec::load_array(
        d,
        &data[row_stride * (57 % D::F32Vec::LEN) + (57 / D::F32Vec::LEN)],
    );
    let mut v58 = D::F32Vec::load_array(
        d,
        &data[row_stride * (58 % D::F32Vec::LEN) + (58 / D::F32Vec::LEN)],
    );
    let mut v59 = D::F32Vec::load_array(
        d,
        &data[row_stride * (59 % D::F32Vec::LEN) + (59 / D::F32Vec::LEN)],
    );
    let mut v60 = D::F32Vec::load_array(
        d,
        &data[row_stride * (60 % D::F32Vec::LEN) + (60 / D::F32Vec::LEN)],
    );
    let mut v61 = D::F32Vec::load_array(
        d,
        &data[row_stride * (61 % D::F32Vec::LEN) + (61 / D::F32Vec::LEN)],
    );
    let mut v62 = D::F32Vec::load_array(
        d,
        &data[row_stride * (62 % D::F32Vec::LEN) + (62 / D::F32Vec::LEN)],
    );
    let mut v63 = D::F32Vec::load_array(
        d,
        &data[row_stride * (63 % D::F32Vec::LEN) + (63 / D::F32Vec::LEN)],
    );
    let mut v64 = D::F32Vec::load_array(
        d,
        &data[row_stride * (64 % D::F32Vec::LEN) + (64 / D::F32Vec::LEN)],
    );
    let mut v65 = D::F32Vec::load_array(
        d,
        &data[row_stride * (65 % D::F32Vec::LEN) + (65 / D::F32Vec::LEN)],
    );
    let mut v66 = D::F32Vec::load_array(
        d,
        &data[row_stride * (66 % D::F32Vec::LEN) + (66 / D::F32Vec::LEN)],
    );
    let mut v67 = D::F32Vec::load_array(
        d,
        &data[row_stride * (67 % D::F32Vec::LEN) + (67 / D::F32Vec::LEN)],
    );
    let mut v68 = D::F32Vec::load_array(
        d,
        &data[row_stride * (68 % D::F32Vec::LEN) + (68 / D::F32Vec::LEN)],
    );
    let mut v69 = D::F32Vec::load_array(
        d,
        &data[row_stride * (69 % D::F32Vec::LEN) + (69 / D::F32Vec::LEN)],
    );
    let mut v70 = D::F32Vec::load_array(
        d,
        &data[row_stride * (70 % D::F32Vec::LEN) + (70 / D::F32Vec::LEN)],
    );
    let mut v71 = D::F32Vec::load_array(
        d,
        &data[row_stride * (71 % D::F32Vec::LEN) + (71 / D::F32Vec::LEN)],
    );
    let mut v72 = D::F32Vec::load_array(
        d,
        &data[row_stride * (72 % D::F32Vec::LEN) + (72 / D::F32Vec::LEN)],
    );
    let mut v73 = D::F32Vec::load_array(
        d,
        &data[row_stride * (73 % D::F32Vec::LEN) + (73 / D::F32Vec::LEN)],
    );
    let mut v74 = D::F32Vec::load_array(
        d,
        &data[row_stride * (74 % D::F32Vec::LEN) + (74 / D::F32Vec::LEN)],
    );
    let mut v75 = D::F32Vec::load_array(
        d,
        &data[row_stride * (75 % D::F32Vec::LEN) + (75 / D::F32Vec::LEN)],
    );
    let mut v76 = D::F32Vec::load_array(
        d,
        &data[row_stride * (76 % D::F32Vec::LEN) + (76 / D::F32Vec::LEN)],
    );
    let mut v77 = D::F32Vec::load_array(
        d,
        &data[row_stride * (77 % D::F32Vec::LEN) + (77 / D::F32Vec::LEN)],
    );
    let mut v78 = D::F32Vec::load_array(
        d,
        &data[row_stride * (78 % D::F32Vec::LEN) + (78 / D::F32Vec::LEN)],
    );
    let mut v79 = D::F32Vec::load_array(
        d,
        &data[row_stride * (79 % D::F32Vec::LEN) + (79 / D::F32Vec::LEN)],
    );
    let mut v80 = D::F32Vec::load_array(
        d,
        &data[row_stride * (80 % D::F32Vec::LEN) + (80 / D::F32Vec::LEN)],
    );
    let mut v81 = D::F32Vec::load_array(
        d,
        &data[row_stride * (81 % D::F32Vec::LEN) + (81 / D::F32Vec::LEN)],
    );
    let mut v82 = D::F32Vec::load_array(
        d,
        &data[row_stride * (82 % D::F32Vec::LEN) + (82 / D::F32Vec::LEN)],
    );
    let mut v83 = D::F32Vec::load_array(
        d,
        &data[row_stride * (83 % D::F32Vec::LEN) + (83 / D::F32Vec::LEN)],
    );
    let mut v84 = D::F32Vec::load_array(
        d,
        &data[row_stride * (84 % D::F32Vec::LEN) + (84 / D::F32Vec::LEN)],
    );
    let mut v85 = D::F32Vec::load_array(
        d,
        &data[row_stride * (85 % D::F32Vec::LEN) + (85 / D::F32Vec::LEN)],
    );
    let mut v86 = D::F32Vec::load_array(
        d,
        &data[row_stride * (86 % D::F32Vec::LEN) + (86 / D::F32Vec::LEN)],
    );
    let mut v87 = D::F32Vec::load_array(
        d,
        &data[row_stride * (87 % D::F32Vec::LEN) + (87 / D::F32Vec::LEN)],
    );
    let mut v88 = D::F32Vec::load_array(
        d,
        &data[row_stride * (88 % D::F32Vec::LEN) + (88 / D::F32Vec::LEN)],
    );
    let mut v89 = D::F32Vec::load_array(
        d,
        &data[row_stride * (89 % D::F32Vec::LEN) + (89 / D::F32Vec::LEN)],
    );
    let mut v90 = D::F32Vec::load_array(
        d,
        &data[row_stride * (90 % D::F32Vec::LEN) + (90 / D::F32Vec::LEN)],
    );
    let mut v91 = D::F32Vec::load_array(
        d,
        &data[row_stride * (91 % D::F32Vec::LEN) + (91 / D::F32Vec::LEN)],
    );
    let mut v92 = D::F32Vec::load_array(
        d,
        &data[row_stride * (92 % D::F32Vec::LEN) + (92 / D::F32Vec::LEN)],
    );
    let mut v93 = D::F32Vec::load_array(
        d,
        &data[row_stride * (93 % D::F32Vec::LEN) + (93 / D::F32Vec::LEN)],
    );
    let mut v94 = D::F32Vec::load_array(
        d,
        &data[row_stride * (94 % D::F32Vec::LEN) + (94 / D::F32Vec::LEN)],
    );
    let mut v95 = D::F32Vec::load_array(
        d,
        &data[row_stride * (95 % D::F32Vec::LEN) + (95 / D::F32Vec::LEN)],
    );
    let mut v96 = D::F32Vec::load_array(
        d,
        &data[row_stride * (96 % D::F32Vec::LEN) + (96 / D::F32Vec::LEN)],
    );
    let mut v97 = D::F32Vec::load_array(
        d,
        &data[row_stride * (97 % D::F32Vec::LEN) + (97 / D::F32Vec::LEN)],
    );
    let mut v98 = D::F32Vec::load_array(
        d,
        &data[row_stride * (98 % D::F32Vec::LEN) + (98 / D::F32Vec::LEN)],
    );
    let mut v99 = D::F32Vec::load_array(
        d,
        &data[row_stride * (99 % D::F32Vec::LEN) + (99 / D::F32Vec::LEN)],
    );
    let mut v100 = D::F32Vec::load_array(
        d,
        &data[row_stride * (100 % D::F32Vec::LEN) + (100 / D::F32Vec::LEN)],
    );
    let mut v101 = D::F32Vec::load_array(
        d,
        &data[row_stride * (101 % D::F32Vec::LEN) + (101 / D::F32Vec::LEN)],
    );
    let mut v102 = D::F32Vec::load_array(
        d,
        &data[row_stride * (102 % D::F32Vec::LEN) + (102 / D::F32Vec::LEN)],
    );
    let mut v103 = D::F32Vec::load_array(
        d,
        &data[row_stride * (103 % D::F32Vec::LEN) + (103 / D::F32Vec::LEN)],
    );
    let mut v104 = D::F32Vec::load_array(
        d,
        &data[row_stride * (104 % D::F32Vec::LEN) + (104 / D::F32Vec::LEN)],
    );
    let mut v105 = D::F32Vec::load_array(
        d,
        &data[row_stride * (105 % D::F32Vec::LEN) + (105 / D::F32Vec::LEN)],
    );
    let mut v106 = D::F32Vec::load_array(
        d,
        &data[row_stride * (106 % D::F32Vec::LEN) + (106 / D::F32Vec::LEN)],
    );
    let mut v107 = D::F32Vec::load_array(
        d,
        &data[row_stride * (107 % D::F32Vec::LEN) + (107 / D::F32Vec::LEN)],
    );
    let mut v108 = D::F32Vec::load_array(
        d,
        &data[row_stride * (108 % D::F32Vec::LEN) + (108 / D::F32Vec::LEN)],
    );
    let mut v109 = D::F32Vec::load_array(
        d,
        &data[row_stride * (109 % D::F32Vec::LEN) + (109 / D::F32Vec::LEN)],
    );
    let mut v110 = D::F32Vec::load_array(
        d,
        &data[row_stride * (110 % D::F32Vec::LEN) + (110 / D::F32Vec::LEN)],
    );
    let mut v111 = D::F32Vec::load_array(
        d,
        &data[row_stride * (111 % D::F32Vec::LEN) + (111 / D::F32Vec::LEN)],
    );
    let mut v112 = D::F32Vec::load_array(
        d,
        &data[row_stride * (112 % D::F32Vec::LEN) + (112 / D::F32Vec::LEN)],
    );
    let mut v113 = D::F32Vec::load_array(
        d,
        &data[row_stride * (113 % D::F32Vec::LEN) + (113 / D::F32Vec::LEN)],
    );
    let mut v114 = D::F32Vec::load_array(
        d,
        &data[row_stride * (114 % D::F32Vec::LEN) + (114 / D::F32Vec::LEN)],
    );
    let mut v115 = D::F32Vec::load_array(
        d,
        &data[row_stride * (115 % D::F32Vec::LEN) + (115 / D::F32Vec::LEN)],
    );
    let mut v116 = D::F32Vec::load_array(
        d,
        &data[row_stride * (116 % D::F32Vec::LEN) + (116 / D::F32Vec::LEN)],
    );
    let mut v117 = D::F32Vec::load_array(
        d,
        &data[row_stride * (117 % D::F32Vec::LEN) + (117 / D::F32Vec::LEN)],
    );
    let mut v118 = D::F32Vec::load_array(
        d,
        &data[row_stride * (118 % D::F32Vec::LEN) + (118 / D::F32Vec::LEN)],
    );
    let mut v119 = D::F32Vec::load_array(
        d,
        &data[row_stride * (119 % D::F32Vec::LEN) + (119 / D::F32Vec::LEN)],
    );
    let mut v120 = D::F32Vec::load_array(
        d,
        &data[row_stride * (120 % D::F32Vec::LEN) + (120 / D::F32Vec::LEN)],
    );
    let mut v121 = D::F32Vec::load_array(
        d,
        &data[row_stride * (121 % D::F32Vec::LEN) + (121 / D::F32Vec::LEN)],
    );
    let mut v122 = D::F32Vec::load_array(
        d,
        &data[row_stride * (122 % D::F32Vec::LEN) + (122 / D::F32Vec::LEN)],
    );
    let mut v123 = D::F32Vec::load_array(
        d,
        &data[row_stride * (123 % D::F32Vec::LEN) + (123 / D::F32Vec::LEN)],
    );
    let mut v124 = D::F32Vec::load_array(
        d,
        &data[row_stride * (124 % D::F32Vec::LEN) + (124 / D::F32Vec::LEN)],
    );
    let mut v125 = D::F32Vec::load_array(
        d,
        &data[row_stride * (125 % D::F32Vec::LEN) + (125 / D::F32Vec::LEN)],
    );
    let mut v126 = D::F32Vec::load_array(
        d,
        &data[row_stride * (126 % D::F32Vec::LEN) + (126 / D::F32Vec::LEN)],
    );
    let mut v127 = D::F32Vec::load_array(
        d,
        &data[row_stride * (127 % D::F32Vec::LEN) + (127 / D::F32Vec::LEN)],
    );
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73,
        v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91,
        v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107,
        v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122,
        v123, v124, v125, v126, v127,
    ) = idct_128(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72,
        v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90,
        v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106,
        v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121,
        v122, v123, v124, v125, v126, v127,
    );
    v0.store_array(&mut data[row_stride * (0 % D::F32Vec::LEN) + (0 / D::F32Vec::LEN)]);
    v1.store_array(&mut data[row_stride * (1 % D::F32Vec::LEN) + (1 / D::F32Vec::LEN)]);
    v2.store_array(&mut data[row_stride * (2 % D::F32Vec::LEN) + (2 / D::F32Vec::LEN)]);
    v3.store_array(&mut data[row_stride * (3 % D::F32Vec::LEN) + (3 / D::F32Vec::LEN)]);
    v4.store_array(&mut data[row_stride * (4 % D::F32Vec::LEN) + (4 / D::F32Vec::LEN)]);
    v5.store_array(&mut data[row_stride * (5 % D::F32Vec::LEN) + (5 / D::F32Vec::LEN)]);
    v6.store_array(&mut data[row_stride * (6 % D::F32Vec::LEN) + (6 / D::F32Vec::LEN)]);
    v7.store_array(&mut data[row_stride * (7 % D::F32Vec::LEN) + (7 / D::F32Vec::LEN)]);
    v8.store_array(&mut data[row_stride * (8 % D::F32Vec::LEN) + (8 / D::F32Vec::LEN)]);
    v9.store_array(&mut data[row_stride * (9 % D::F32Vec::LEN) + (9 / D::F32Vec::LEN)]);
    v10.store_array(&mut data[row_stride * (10 % D::F32Vec::LEN) + (10 / D::F32Vec::LEN)]);
    v11.store_array(&mut data[row_stride * (11 % D::F32Vec::LEN) + (11 / D::F32Vec::LEN)]);
    v12.store_array(&mut data[row_stride * (12 % D::F32Vec::LEN) + (12 / D::F32Vec::LEN)]);
    v13.store_array(&mut data[row_stride * (13 % D::F32Vec::LEN) + (13 / D::F32Vec::LEN)]);
    v14.store_array(&mut data[row_stride * (14 % D::F32Vec::LEN) + (14 / D::F32Vec::LEN)]);
    v15.store_array(&mut data[row_stride * (15 % D::F32Vec::LEN) + (15 / D::F32Vec::LEN)]);
    v16.store_array(&mut data[row_stride * (16 % D::F32Vec::LEN) + (16 / D::F32Vec::LEN)]);
    v17.store_array(&mut data[row_stride * (17 % D::F32Vec::LEN) + (17 / D::F32Vec::LEN)]);
    v18.store_array(&mut data[row_stride * (18 % D::F32Vec::LEN) + (18 / D::F32Vec::LEN)]);
    v19.store_array(&mut data[row_stride * (19 % D::F32Vec::LEN) + (19 / D::F32Vec::LEN)]);
    v20.store_array(&mut data[row_stride * (20 % D::F32Vec::LEN) + (20 / D::F32Vec::LEN)]);
    v21.store_array(&mut data[row_stride * (21 % D::F32Vec::LEN) + (21 / D::F32Vec::LEN)]);
    v22.store_array(&mut data[row_stride * (22 % D::F32Vec::LEN) + (22 / D::F32Vec::LEN)]);
    v23.store_array(&mut data[row_stride * (23 % D::F32Vec::LEN) + (23 / D::F32Vec::LEN)]);
    v24.store_array(&mut data[row_stride * (24 % D::F32Vec::LEN) + (24 / D::F32Vec::LEN)]);
    v25.store_array(&mut data[row_stride * (25 % D::F32Vec::LEN) + (25 / D::F32Vec::LEN)]);
    v26.store_array(&mut data[row_stride * (26 % D::F32Vec::LEN) + (26 / D::F32Vec::LEN)]);
    v27.store_array(&mut data[row_stride * (27 % D::F32Vec::LEN) + (27 / D::F32Vec::LEN)]);
    v28.store_array(&mut data[row_stride * (28 % D::F32Vec::LEN) + (28 / D::F32Vec::LEN)]);
    v29.store_array(&mut data[row_stride * (29 % D::F32Vec::LEN) + (29 / D::F32Vec::LEN)]);
    v30.store_array(&mut data[row_stride * (30 % D::F32Vec::LEN) + (30 / D::F32Vec::LEN)]);
    v31.store_array(&mut data[row_stride * (31 % D::F32Vec::LEN) + (31 / D::F32Vec::LEN)]);
    v32.store_array(&mut data[row_stride * (32 % D::F32Vec::LEN) + (32 / D::F32Vec::LEN)]);
    v33.store_array(&mut data[row_stride * (33 % D::F32Vec::LEN) + (33 / D::F32Vec::LEN)]);
    v34.store_array(&mut data[row_stride * (34 % D::F32Vec::LEN) + (34 / D::F32Vec::LEN)]);
    v35.store_array(&mut data[row_stride * (35 % D::F32Vec::LEN) + (35 / D::F32Vec::LEN)]);
    v36.store_array(&mut data[row_stride * (36 % D::F32Vec::LEN) + (36 / D::F32Vec::LEN)]);
    v37.store_array(&mut data[row_stride * (37 % D::F32Vec::LEN) + (37 / D::F32Vec::LEN)]);
    v38.store_array(&mut data[row_stride * (38 % D::F32Vec::LEN) + (38 / D::F32Vec::LEN)]);
    v39.store_array(&mut data[row_stride * (39 % D::F32Vec::LEN) + (39 / D::F32Vec::LEN)]);
    v40.store_array(&mut data[row_stride * (40 % D::F32Vec::LEN) + (40 / D::F32Vec::LEN)]);
    v41.store_array(&mut data[row_stride * (41 % D::F32Vec::LEN) + (41 / D::F32Vec::LEN)]);
    v42.store_array(&mut data[row_stride * (42 % D::F32Vec::LEN) + (42 / D::F32Vec::LEN)]);
    v43.store_array(&mut data[row_stride * (43 % D::F32Vec::LEN) + (43 / D::F32Vec::LEN)]);
    v44.store_array(&mut data[row_stride * (44 % D::F32Vec::LEN) + (44 / D::F32Vec::LEN)]);
    v45.store_array(&mut data[row_stride * (45 % D::F32Vec::LEN) + (45 / D::F32Vec::LEN)]);
    v46.store_array(&mut data[row_stride * (46 % D::F32Vec::LEN) + (46 / D::F32Vec::LEN)]);
    v47.store_array(&mut data[row_stride * (47 % D::F32Vec::LEN) + (47 / D::F32Vec::LEN)]);
    v48.store_array(&mut data[row_stride * (48 % D::F32Vec::LEN) + (48 / D::F32Vec::LEN)]);
    v49.store_array(&mut data[row_stride * (49 % D::F32Vec::LEN) + (49 / D::F32Vec::LEN)]);
    v50.store_array(&mut data[row_stride * (50 % D::F32Vec::LEN) + (50 / D::F32Vec::LEN)]);
    v51.store_array(&mut data[row_stride * (51 % D::F32Vec::LEN) + (51 / D::F32Vec::LEN)]);
    v52.store_array(&mut data[row_stride * (52 % D::F32Vec::LEN) + (52 / D::F32Vec::LEN)]);
    v53.store_array(&mut data[row_stride * (53 % D::F32Vec::LEN) + (53 / D::F32Vec::LEN)]);
    v54.store_array(&mut data[row_stride * (54 % D::F32Vec::LEN) + (54 / D::F32Vec::LEN)]);
    v55.store_array(&mut data[row_stride * (55 % D::F32Vec::LEN) + (55 / D::F32Vec::LEN)]);
    v56.store_array(&mut data[row_stride * (56 % D::F32Vec::LEN) + (56 / D::F32Vec::LEN)]);
    v57.store_array(&mut data[row_stride * (57 % D::F32Vec::LEN) + (57 / D::F32Vec::LEN)]);
    v58.store_array(&mut data[row_stride * (58 % D::F32Vec::LEN) + (58 / D::F32Vec::LEN)]);
    v59.store_array(&mut data[row_stride * (59 % D::F32Vec::LEN) + (59 / D::F32Vec::LEN)]);
    v60.store_array(&mut data[row_stride * (60 % D::F32Vec::LEN) + (60 / D::F32Vec::LEN)]);
    v61.store_array(&mut data[row_stride * (61 % D::F32Vec::LEN) + (61 / D::F32Vec::LEN)]);
    v62.store_array(&mut data[row_stride * (62 % D::F32Vec::LEN) + (62 / D::F32Vec::LEN)]);
    v63.store_array(&mut data[row_stride * (63 % D::F32Vec::LEN) + (63 / D::F32Vec::LEN)]);
    v64.store_array(&mut data[row_stride * (64 % D::F32Vec::LEN) + (64 / D::F32Vec::LEN)]);
    v65.store_array(&mut data[row_stride * (65 % D::F32Vec::LEN) + (65 / D::F32Vec::LEN)]);
    v66.store_array(&mut data[row_stride * (66 % D::F32Vec::LEN) + (66 / D::F32Vec::LEN)]);
    v67.store_array(&mut data[row_stride * (67 % D::F32Vec::LEN) + (67 / D::F32Vec::LEN)]);
    v68.store_array(&mut data[row_stride * (68 % D::F32Vec::LEN) + (68 / D::F32Vec::LEN)]);
    v69.store_array(&mut data[row_stride * (69 % D::F32Vec::LEN) + (69 / D::F32Vec::LEN)]);
    v70.store_array(&mut data[row_stride * (70 % D::F32Vec::LEN) + (70 / D::F32Vec::LEN)]);
    v71.store_array(&mut data[row_stride * (71 % D::F32Vec::LEN) + (71 / D::F32Vec::LEN)]);
    v72.store_array(&mut data[row_stride * (72 % D::F32Vec::LEN) + (72 / D::F32Vec::LEN)]);
    v73.store_array(&mut data[row_stride * (73 % D::F32Vec::LEN) + (73 / D::F32Vec::LEN)]);
    v74.store_array(&mut data[row_stride * (74 % D::F32Vec::LEN) + (74 / D::F32Vec::LEN)]);
    v75.store_array(&mut data[row_stride * (75 % D::F32Vec::LEN) + (75 / D::F32Vec::LEN)]);
    v76.store_array(&mut data[row_stride * (76 % D::F32Vec::LEN) + (76 / D::F32Vec::LEN)]);
    v77.store_array(&mut data[row_stride * (77 % D::F32Vec::LEN) + (77 / D::F32Vec::LEN)]);
    v78.store_array(&mut data[row_stride * (78 % D::F32Vec::LEN) + (78 / D::F32Vec::LEN)]);
    v79.store_array(&mut data[row_stride * (79 % D::F32Vec::LEN) + (79 / D::F32Vec::LEN)]);
    v80.store_array(&mut data[row_stride * (80 % D::F32Vec::LEN) + (80 / D::F32Vec::LEN)]);
    v81.store_array(&mut data[row_stride * (81 % D::F32Vec::LEN) + (81 / D::F32Vec::LEN)]);
    v82.store_array(&mut data[row_stride * (82 % D::F32Vec::LEN) + (82 / D::F32Vec::LEN)]);
    v83.store_array(&mut data[row_stride * (83 % D::F32Vec::LEN) + (83 / D::F32Vec::LEN)]);
    v84.store_array(&mut data[row_stride * (84 % D::F32Vec::LEN) + (84 / D::F32Vec::LEN)]);
    v85.store_array(&mut data[row_stride * (85 % D::F32Vec::LEN) + (85 / D::F32Vec::LEN)]);
    v86.store_array(&mut data[row_stride * (86 % D::F32Vec::LEN) + (86 / D::F32Vec::LEN)]);
    v87.store_array(&mut data[row_stride * (87 % D::F32Vec::LEN) + (87 / D::F32Vec::LEN)]);
    v88.store_array(&mut data[row_stride * (88 % D::F32Vec::LEN) + (88 / D::F32Vec::LEN)]);
    v89.store_array(&mut data[row_stride * (89 % D::F32Vec::LEN) + (89 / D::F32Vec::LEN)]);
    v90.store_array(&mut data[row_stride * (90 % D::F32Vec::LEN) + (90 / D::F32Vec::LEN)]);
    v91.store_array(&mut data[row_stride * (91 % D::F32Vec::LEN) + (91 / D::F32Vec::LEN)]);
    v92.store_array(&mut data[row_stride * (92 % D::F32Vec::LEN) + (92 / D::F32Vec::LEN)]);
    v93.store_array(&mut data[row_stride * (93 % D::F32Vec::LEN) + (93 / D::F32Vec::LEN)]);
    v94.store_array(&mut data[row_stride * (94 % D::F32Vec::LEN) + (94 / D::F32Vec::LEN)]);
    v95.store_array(&mut data[row_stride * (95 % D::F32Vec::LEN) + (95 / D::F32Vec::LEN)]);
    v96.store_array(&mut data[row_stride * (96 % D::F32Vec::LEN) + (96 / D::F32Vec::LEN)]);
    v97.store_array(&mut data[row_stride * (97 % D::F32Vec::LEN) + (97 / D::F32Vec::LEN)]);
    v98.store_array(&mut data[row_stride * (98 % D::F32Vec::LEN) + (98 / D::F32Vec::LEN)]);
    v99.store_array(&mut data[row_stride * (99 % D::F32Vec::LEN) + (99 / D::F32Vec::LEN)]);
    v100.store_array(&mut data[row_stride * (100 % D::F32Vec::LEN) + (100 / D::F32Vec::LEN)]);
    v101.store_array(&mut data[row_stride * (101 % D::F32Vec::LEN) + (101 / D::F32Vec::LEN)]);
    v102.store_array(&mut data[row_stride * (102 % D::F32Vec::LEN) + (102 / D::F32Vec::LEN)]);
    v103.store_array(&mut data[row_stride * (103 % D::F32Vec::LEN) + (103 / D::F32Vec::LEN)]);
    v104.store_array(&mut data[row_stride * (104 % D::F32Vec::LEN) + (104 / D::F32Vec::LEN)]);
    v105.store_array(&mut data[row_stride * (105 % D::F32Vec::LEN) + (105 / D::F32Vec::LEN)]);
    v106.store_array(&mut data[row_stride * (106 % D::F32Vec::LEN) + (106 / D::F32Vec::LEN)]);
    v107.store_array(&mut data[row_stride * (107 % D::F32Vec::LEN) + (107 / D::F32Vec::LEN)]);
    v108.store_array(&mut data[row_stride * (108 % D::F32Vec::LEN) + (108 / D::F32Vec::LEN)]);
    v109.store_array(&mut data[row_stride * (109 % D::F32Vec::LEN) + (109 / D::F32Vec::LEN)]);
    v110.store_array(&mut data[row_stride * (110 % D::F32Vec::LEN) + (110 / D::F32Vec::LEN)]);
    v111.store_array(&mut data[row_stride * (111 % D::F32Vec::LEN) + (111 / D::F32Vec::LEN)]);
    v112.store_array(&mut data[row_stride * (112 % D::F32Vec::LEN) + (112 / D::F32Vec::LEN)]);
    v113.store_array(&mut data[row_stride * (113 % D::F32Vec::LEN) + (113 / D::F32Vec::LEN)]);
    v114.store_array(&mut data[row_stride * (114 % D::F32Vec::LEN) + (114 / D::F32Vec::LEN)]);
    v115.store_array(&mut data[row_stride * (115 % D::F32Vec::LEN) + (115 / D::F32Vec::LEN)]);
    v116.store_array(&mut data[row_stride * (116 % D::F32Vec::LEN) + (116 / D::F32Vec::LEN)]);
    v117.store_array(&mut data[row_stride * (117 % D::F32Vec::LEN) + (117 / D::F32Vec::LEN)]);
    v118.store_array(&mut data[row_stride * (118 % D::F32Vec::LEN) + (118 / D::F32Vec::LEN)]);
    v119.store_array(&mut data[row_stride * (119 % D::F32Vec::LEN) + (119 / D::F32Vec::LEN)]);
    v120.store_array(&mut data[row_stride * (120 % D::F32Vec::LEN) + (120 / D::F32Vec::LEN)]);
    v121.store_array(&mut data[row_stride * (121 % D::F32Vec::LEN) + (121 / D::F32Vec::LEN)]);
    v122.store_array(&mut data[row_stride * (122 % D::F32Vec::LEN) + (122 / D::F32Vec::LEN)]);
    v123.store_array(&mut data[row_stride * (123 % D::F32Vec::LEN) + (123 / D::F32Vec::LEN)]);
    v124.store_array(&mut data[row_stride * (124 % D::F32Vec::LEN) + (124 / D::F32Vec::LEN)]);
    v125.store_array(&mut data[row_stride * (125 % D::F32Vec::LEN) + (125 / D::F32Vec::LEN)]);
    v126.store_array(&mut data[row_stride * (126 % D::F32Vec::LEN) + (126 / D::F32Vec::LEN)]);
    v127.store_array(&mut data[row_stride * (127 % D::F32Vec::LEN) + (127 / D::F32Vec::LEN)]);
}

#[inline(always)]
pub(super) fn do_idct_128_trh<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
) {
    let row_stride = 64 / D::F32Vec::LEN;
    assert!(data.len() > 127 * row_stride);
    const { assert!(64usize.is_multiple_of(D::F32Vec::LEN)) };
    let mut v0 = D::F32Vec::load_array(d, &data[row_stride * 0]);
    let mut v1 = D::F32Vec::load_array(d, &data[row_stride * 2]);
    let mut v2 = D::F32Vec::load_array(d, &data[row_stride * 4]);
    let mut v3 = D::F32Vec::load_array(d, &data[row_stride * 6]);
    let mut v4 = D::F32Vec::load_array(d, &data[row_stride * 8]);
    let mut v5 = D::F32Vec::load_array(d, &data[row_stride * 10]);
    let mut v6 = D::F32Vec::load_array(d, &data[row_stride * 12]);
    let mut v7 = D::F32Vec::load_array(d, &data[row_stride * 14]);
    let mut v8 = D::F32Vec::load_array(d, &data[row_stride * 16]);
    let mut v9 = D::F32Vec::load_array(d, &data[row_stride * 18]);
    let mut v10 = D::F32Vec::load_array(d, &data[row_stride * 20]);
    let mut v11 = D::F32Vec::load_array(d, &data[row_stride * 22]);
    let mut v12 = D::F32Vec::load_array(d, &data[row_stride * 24]);
    let mut v13 = D::F32Vec::load_array(d, &data[row_stride * 26]);
    let mut v14 = D::F32Vec::load_array(d, &data[row_stride * 28]);
    let mut v15 = D::F32Vec::load_array(d, &data[row_stride * 30]);
    let mut v16 = D::F32Vec::load_array(d, &data[row_stride * 32]);
    let mut v17 = D::F32Vec::load_array(d, &data[row_stride * 34]);
    let mut v18 = D::F32Vec::load_array(d, &data[row_stride * 36]);
    let mut v19 = D::F32Vec::load_array(d, &data[row_stride * 38]);
    let mut v20 = D::F32Vec::load_array(d, &data[row_stride * 40]);
    let mut v21 = D::F32Vec::load_array(d, &data[row_stride * 42]);
    let mut v22 = D::F32Vec::load_array(d, &data[row_stride * 44]);
    let mut v23 = D::F32Vec::load_array(d, &data[row_stride * 46]);
    let mut v24 = D::F32Vec::load_array(d, &data[row_stride * 48]);
    let mut v25 = D::F32Vec::load_array(d, &data[row_stride * 50]);
    let mut v26 = D::F32Vec::load_array(d, &data[row_stride * 52]);
    let mut v27 = D::F32Vec::load_array(d, &data[row_stride * 54]);
    let mut v28 = D::F32Vec::load_array(d, &data[row_stride * 56]);
    let mut v29 = D::F32Vec::load_array(d, &data[row_stride * 58]);
    let mut v30 = D::F32Vec::load_array(d, &data[row_stride * 60]);
    let mut v31 = D::F32Vec::load_array(d, &data[row_stride * 62]);
    let mut v32 = D::F32Vec::load_array(d, &data[row_stride * 64]);
    let mut v33 = D::F32Vec::load_array(d, &data[row_stride * 66]);
    let mut v34 = D::F32Vec::load_array(d, &data[row_stride * 68]);
    let mut v35 = D::F32Vec::load_array(d, &data[row_stride * 70]);
    let mut v36 = D::F32Vec::load_array(d, &data[row_stride * 72]);
    let mut v37 = D::F32Vec::load_array(d, &data[row_stride * 74]);
    let mut v38 = D::F32Vec::load_array(d, &data[row_stride * 76]);
    let mut v39 = D::F32Vec::load_array(d, &data[row_stride * 78]);
    let mut v40 = D::F32Vec::load_array(d, &data[row_stride * 80]);
    let mut v41 = D::F32Vec::load_array(d, &data[row_stride * 82]);
    let mut v42 = D::F32Vec::load_array(d, &data[row_stride * 84]);
    let mut v43 = D::F32Vec::load_array(d, &data[row_stride * 86]);
    let mut v44 = D::F32Vec::load_array(d, &data[row_stride * 88]);
    let mut v45 = D::F32Vec::load_array(d, &data[row_stride * 90]);
    let mut v46 = D::F32Vec::load_array(d, &data[row_stride * 92]);
    let mut v47 = D::F32Vec::load_array(d, &data[row_stride * 94]);
    let mut v48 = D::F32Vec::load_array(d, &data[row_stride * 96]);
    let mut v49 = D::F32Vec::load_array(d, &data[row_stride * 98]);
    let mut v50 = D::F32Vec::load_array(d, &data[row_stride * 100]);
    let mut v51 = D::F32Vec::load_array(d, &data[row_stride * 102]);
    let mut v52 = D::F32Vec::load_array(d, &data[row_stride * 104]);
    let mut v53 = D::F32Vec::load_array(d, &data[row_stride * 106]);
    let mut v54 = D::F32Vec::load_array(d, &data[row_stride * 108]);
    let mut v55 = D::F32Vec::load_array(d, &data[row_stride * 110]);
    let mut v56 = D::F32Vec::load_array(d, &data[row_stride * 112]);
    let mut v57 = D::F32Vec::load_array(d, &data[row_stride * 114]);
    let mut v58 = D::F32Vec::load_array(d, &data[row_stride * 116]);
    let mut v59 = D::F32Vec::load_array(d, &data[row_stride * 118]);
    let mut v60 = D::F32Vec::load_array(d, &data[row_stride * 120]);
    let mut v61 = D::F32Vec::load_array(d, &data[row_stride * 122]);
    let mut v62 = D::F32Vec::load_array(d, &data[row_stride * 124]);
    let mut v63 = D::F32Vec::load_array(d, &data[row_stride * 126]);
    let mut v64 = D::F32Vec::load_array(d, &data[row_stride * 1]);
    let mut v65 = D::F32Vec::load_array(d, &data[row_stride * 3]);
    let mut v66 = D::F32Vec::load_array(d, &data[row_stride * 5]);
    let mut v67 = D::F32Vec::load_array(d, &data[row_stride * 7]);
    let mut v68 = D::F32Vec::load_array(d, &data[row_stride * 9]);
    let mut v69 = D::F32Vec::load_array(d, &data[row_stride * 11]);
    let mut v70 = D::F32Vec::load_array(d, &data[row_stride * 13]);
    let mut v71 = D::F32Vec::load_array(d, &data[row_stride * 15]);
    let mut v72 = D::F32Vec::load_array(d, &data[row_stride * 17]);
    let mut v73 = D::F32Vec::load_array(d, &data[row_stride * 19]);
    let mut v74 = D::F32Vec::load_array(d, &data[row_stride * 21]);
    let mut v75 = D::F32Vec::load_array(d, &data[row_stride * 23]);
    let mut v76 = D::F32Vec::load_array(d, &data[row_stride * 25]);
    let mut v77 = D::F32Vec::load_array(d, &data[row_stride * 27]);
    let mut v78 = D::F32Vec::load_array(d, &data[row_stride * 29]);
    let mut v79 = D::F32Vec::load_array(d, &data[row_stride * 31]);
    let mut v80 = D::F32Vec::load_array(d, &data[row_stride * 33]);
    let mut v81 = D::F32Vec::load_array(d, &data[row_stride * 35]);
    let mut v82 = D::F32Vec::load_array(d, &data[row_stride * 37]);
    let mut v83 = D::F32Vec::load_array(d, &data[row_stride * 39]);
    let mut v84 = D::F32Vec::load_array(d, &data[row_stride * 41]);
    let mut v85 = D::F32Vec::load_array(d, &data[row_stride * 43]);
    let mut v86 = D::F32Vec::load_array(d, &data[row_stride * 45]);
    let mut v87 = D::F32Vec::load_array(d, &data[row_stride * 47]);
    let mut v88 = D::F32Vec::load_array(d, &data[row_stride * 49]);
    let mut v89 = D::F32Vec::load_array(d, &data[row_stride * 51]);
    let mut v90 = D::F32Vec::load_array(d, &data[row_stride * 53]);
    let mut v91 = D::F32Vec::load_array(d, &data[row_stride * 55]);
    let mut v92 = D::F32Vec::load_array(d, &data[row_stride * 57]);
    let mut v93 = D::F32Vec::load_array(d, &data[row_stride * 59]);
    let mut v94 = D::F32Vec::load_array(d, &data[row_stride * 61]);
    let mut v95 = D::F32Vec::load_array(d, &data[row_stride * 63]);
    let mut v96 = D::F32Vec::load_array(d, &data[row_stride * 65]);
    let mut v97 = D::F32Vec::load_array(d, &data[row_stride * 67]);
    let mut v98 = D::F32Vec::load_array(d, &data[row_stride * 69]);
    let mut v99 = D::F32Vec::load_array(d, &data[row_stride * 71]);
    let mut v100 = D::F32Vec::load_array(d, &data[row_stride * 73]);
    let mut v101 = D::F32Vec::load_array(d, &data[row_stride * 75]);
    let mut v102 = D::F32Vec::load_array(d, &data[row_stride * 77]);
    let mut v103 = D::F32Vec::load_array(d, &data[row_stride * 79]);
    let mut v104 = D::F32Vec::load_array(d, &data[row_stride * 81]);
    let mut v105 = D::F32Vec::load_array(d, &data[row_stride * 83]);
    let mut v106 = D::F32Vec::load_array(d, &data[row_stride * 85]);
    let mut v107 = D::F32Vec::load_array(d, &data[row_stride * 87]);
    let mut v108 = D::F32Vec::load_array(d, &data[row_stride * 89]);
    let mut v109 = D::F32Vec::load_array(d, &data[row_stride * 91]);
    let mut v110 = D::F32Vec::load_array(d, &data[row_stride * 93]);
    let mut v111 = D::F32Vec::load_array(d, &data[row_stride * 95]);
    let mut v112 = D::F32Vec::load_array(d, &data[row_stride * 97]);
    let mut v113 = D::F32Vec::load_array(d, &data[row_stride * 99]);
    let mut v114 = D::F32Vec::load_array(d, &data[row_stride * 101]);
    let mut v115 = D::F32Vec::load_array(d, &data[row_stride * 103]);
    let mut v116 = D::F32Vec::load_array(d, &data[row_stride * 105]);
    let mut v117 = D::F32Vec::load_array(d, &data[row_stride * 107]);
    let mut v118 = D::F32Vec::load_array(d, &data[row_stride * 109]);
    let mut v119 = D::F32Vec::load_array(d, &data[row_stride * 111]);
    let mut v120 = D::F32Vec::load_array(d, &data[row_stride * 113]);
    let mut v121 = D::F32Vec::load_array(d, &data[row_stride * 115]);
    let mut v122 = D::F32Vec::load_array(d, &data[row_stride * 117]);
    let mut v123 = D::F32Vec::load_array(d, &data[row_stride * 119]);
    let mut v124 = D::F32Vec::load_array(d, &data[row_stride * 121]);
    let mut v125 = D::F32Vec::load_array(d, &data[row_stride * 123]);
    let mut v126 = D::F32Vec::load_array(d, &data[row_stride * 125]);
    let mut v127 = D::F32Vec::load_array(d, &data[row_stride * 127]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73,
        v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91,
        v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107,
        v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122,
        v123, v124, v125, v126, v127,
    ) = idct_128(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72,
        v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90,
        v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106,
        v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121,
        v122, v123, v124, v125, v126, v127,
    );
    v0.store_array(&mut data[row_stride * 0]);
    v1.store_array(&mut data[row_stride * 1]);
    v2.store_array(&mut data[row_stride * 2]);
    v3.store_array(&mut data[row_stride * 3]);
    v4.store_array(&mut data[row_stride * 4]);
    v5.store_array(&mut data[row_stride * 5]);
    v6.store_array(&mut data[row_stride * 6]);
    v7.store_array(&mut data[row_stride * 7]);
    v8.store_array(&mut data[row_stride * 8]);
    v9.store_array(&mut data[row_stride * 9]);
    v10.store_array(&mut data[row_stride * 10]);
    v11.store_array(&mut data[row_stride * 11]);
    v12.store_array(&mut data[row_stride * 12]);
    v13.store_array(&mut data[row_stride * 13]);
    v14.store_array(&mut data[row_stride * 14]);
    v15.store_array(&mut data[row_stride * 15]);
    v16.store_array(&mut data[row_stride * 16]);
    v17.store_array(&mut data[row_stride * 17]);
    v18.store_array(&mut data[row_stride * 18]);
    v19.store_array(&mut data[row_stride * 19]);
    v20.store_array(&mut data[row_stride * 20]);
    v21.store_array(&mut data[row_stride * 21]);
    v22.store_array(&mut data[row_stride * 22]);
    v23.store_array(&mut data[row_stride * 23]);
    v24.store_array(&mut data[row_stride * 24]);
    v25.store_array(&mut data[row_stride * 25]);
    v26.store_array(&mut data[row_stride * 26]);
    v27.store_array(&mut data[row_stride * 27]);
    v28.store_array(&mut data[row_stride * 28]);
    v29.store_array(&mut data[row_stride * 29]);
    v30.store_array(&mut data[row_stride * 30]);
    v31.store_array(&mut data[row_stride * 31]);
    v32.store_array(&mut data[row_stride * 32]);
    v33.store_array(&mut data[row_stride * 33]);
    v34.store_array(&mut data[row_stride * 34]);
    v35.store_array(&mut data[row_stride * 35]);
    v36.store_array(&mut data[row_stride * 36]);
    v37.store_array(&mut data[row_stride * 37]);
    v38.store_array(&mut data[row_stride * 38]);
    v39.store_array(&mut data[row_stride * 39]);
    v40.store_array(&mut data[row_stride * 40]);
    v41.store_array(&mut data[row_stride * 41]);
    v42.store_array(&mut data[row_stride * 42]);
    v43.store_array(&mut data[row_stride * 43]);
    v44.store_array(&mut data[row_stride * 44]);
    v45.store_array(&mut data[row_stride * 45]);
    v46.store_array(&mut data[row_stride * 46]);
    v47.store_array(&mut data[row_stride * 47]);
    v48.store_array(&mut data[row_stride * 48]);
    v49.store_array(&mut data[row_stride * 49]);
    v50.store_array(&mut data[row_stride * 50]);
    v51.store_array(&mut data[row_stride * 51]);
    v52.store_array(&mut data[row_stride * 52]);
    v53.store_array(&mut data[row_stride * 53]);
    v54.store_array(&mut data[row_stride * 54]);
    v55.store_array(&mut data[row_stride * 55]);
    v56.store_array(&mut data[row_stride * 56]);
    v57.store_array(&mut data[row_stride * 57]);
    v58.store_array(&mut data[row_stride * 58]);
    v59.store_array(&mut data[row_stride * 59]);
    v60.store_array(&mut data[row_stride * 60]);
    v61.store_array(&mut data[row_stride * 61]);
    v62.store_array(&mut data[row_stride * 62]);
    v63.store_array(&mut data[row_stride * 63]);
    v64.store_array(&mut data[row_stride * 64]);
    v65.store_array(&mut data[row_stride * 65]);
    v66.store_array(&mut data[row_stride * 66]);
    v67.store_array(&mut data[row_stride * 67]);
    v68.store_array(&mut data[row_stride * 68]);
    v69.store_array(&mut data[row_stride * 69]);
    v70.store_array(&mut data[row_stride * 70]);
    v71.store_array(&mut data[row_stride * 71]);
    v72.store_array(&mut data[row_stride * 72]);
    v73.store_array(&mut data[row_stride * 73]);
    v74.store_array(&mut data[row_stride * 74]);
    v75.store_array(&mut data[row_stride * 75]);
    v76.store_array(&mut data[row_stride * 76]);
    v77.store_array(&mut data[row_stride * 77]);
    v78.store_array(&mut data[row_stride * 78]);
    v79.store_array(&mut data[row_stride * 79]);
    v80.store_array(&mut data[row_stride * 80]);
    v81.store_array(&mut data[row_stride * 81]);
    v82.store_array(&mut data[row_stride * 82]);
    v83.store_array(&mut data[row_stride * 83]);
    v84.store_array(&mut data[row_stride * 84]);
    v85.store_array(&mut data[row_stride * 85]);
    v86.store_array(&mut data[row_stride * 86]);
    v87.store_array(&mut data[row_stride * 87]);
    v88.store_array(&mut data[row_stride * 88]);
    v89.store_array(&mut data[row_stride * 89]);
    v90.store_array(&mut data[row_stride * 90]);
    v91.store_array(&mut data[row_stride * 91]);
    v92.store_array(&mut data[row_stride * 92]);
    v93.store_array(&mut data[row_stride * 93]);
    v94.store_array(&mut data[row_stride * 94]);
    v95.store_array(&mut data[row_stride * 95]);
    v96.store_array(&mut data[row_stride * 96]);
    v97.store_array(&mut data[row_stride * 97]);
    v98.store_array(&mut data[row_stride * 98]);
    v99.store_array(&mut data[row_stride * 99]);
    v100.store_array(&mut data[row_stride * 100]);
    v101.store_array(&mut data[row_stride * 101]);
    v102.store_array(&mut data[row_stride * 102]);
    v103.store_array(&mut data[row_stride * 103]);
    v104.store_array(&mut data[row_stride * 104]);
    v105.store_array(&mut data[row_stride * 105]);
    v106.store_array(&mut data[row_stride * 106]);
    v107.store_array(&mut data[row_stride * 107]);
    v108.store_array(&mut data[row_stride * 108]);
    v109.store_array(&mut data[row_stride * 109]);
    v110.store_array(&mut data[row_stride * 110]);
    v111.store_array(&mut data[row_stride * 111]);
    v112.store_array(&mut data[row_stride * 112]);
    v113.store_array(&mut data[row_stride * 113]);
    v114.store_array(&mut data[row_stride * 114]);
    v115.store_array(&mut data[row_stride * 115]);
    v116.store_array(&mut data[row_stride * 116]);
    v117.store_array(&mut data[row_stride * 117]);
    v118.store_array(&mut data[row_stride * 118]);
    v119.store_array(&mut data[row_stride * 119]);
    v120.store_array(&mut data[row_stride * 120]);
    v121.store_array(&mut data[row_stride * 121]);
    v122.store_array(&mut data[row_stride * 122]);
    v123.store_array(&mut data[row_stride * 123]);
    v124.store_array(&mut data[row_stride * 124]);
    v125.store_array(&mut data[row_stride * 125]);
    v126.store_array(&mut data[row_stride * 126]);
    v127.store_array(&mut data[row_stride * 127]);
}
