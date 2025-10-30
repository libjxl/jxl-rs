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
pub(super) fn idct_256<D: SimdDescriptor>(
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
    mut v128: D::F32Vec,
    mut v129: D::F32Vec,
    mut v130: D::F32Vec,
    mut v131: D::F32Vec,
    mut v132: D::F32Vec,
    mut v133: D::F32Vec,
    mut v134: D::F32Vec,
    mut v135: D::F32Vec,
    mut v136: D::F32Vec,
    mut v137: D::F32Vec,
    mut v138: D::F32Vec,
    mut v139: D::F32Vec,
    mut v140: D::F32Vec,
    mut v141: D::F32Vec,
    mut v142: D::F32Vec,
    mut v143: D::F32Vec,
    mut v144: D::F32Vec,
    mut v145: D::F32Vec,
    mut v146: D::F32Vec,
    mut v147: D::F32Vec,
    mut v148: D::F32Vec,
    mut v149: D::F32Vec,
    mut v150: D::F32Vec,
    mut v151: D::F32Vec,
    mut v152: D::F32Vec,
    mut v153: D::F32Vec,
    mut v154: D::F32Vec,
    mut v155: D::F32Vec,
    mut v156: D::F32Vec,
    mut v157: D::F32Vec,
    mut v158: D::F32Vec,
    mut v159: D::F32Vec,
    mut v160: D::F32Vec,
    mut v161: D::F32Vec,
    mut v162: D::F32Vec,
    mut v163: D::F32Vec,
    mut v164: D::F32Vec,
    mut v165: D::F32Vec,
    mut v166: D::F32Vec,
    mut v167: D::F32Vec,
    mut v168: D::F32Vec,
    mut v169: D::F32Vec,
    mut v170: D::F32Vec,
    mut v171: D::F32Vec,
    mut v172: D::F32Vec,
    mut v173: D::F32Vec,
    mut v174: D::F32Vec,
    mut v175: D::F32Vec,
    mut v176: D::F32Vec,
    mut v177: D::F32Vec,
    mut v178: D::F32Vec,
    mut v179: D::F32Vec,
    mut v180: D::F32Vec,
    mut v181: D::F32Vec,
    mut v182: D::F32Vec,
    mut v183: D::F32Vec,
    mut v184: D::F32Vec,
    mut v185: D::F32Vec,
    mut v186: D::F32Vec,
    mut v187: D::F32Vec,
    mut v188: D::F32Vec,
    mut v189: D::F32Vec,
    mut v190: D::F32Vec,
    mut v191: D::F32Vec,
    mut v192: D::F32Vec,
    mut v193: D::F32Vec,
    mut v194: D::F32Vec,
    mut v195: D::F32Vec,
    mut v196: D::F32Vec,
    mut v197: D::F32Vec,
    mut v198: D::F32Vec,
    mut v199: D::F32Vec,
    mut v200: D::F32Vec,
    mut v201: D::F32Vec,
    mut v202: D::F32Vec,
    mut v203: D::F32Vec,
    mut v204: D::F32Vec,
    mut v205: D::F32Vec,
    mut v206: D::F32Vec,
    mut v207: D::F32Vec,
    mut v208: D::F32Vec,
    mut v209: D::F32Vec,
    mut v210: D::F32Vec,
    mut v211: D::F32Vec,
    mut v212: D::F32Vec,
    mut v213: D::F32Vec,
    mut v214: D::F32Vec,
    mut v215: D::F32Vec,
    mut v216: D::F32Vec,
    mut v217: D::F32Vec,
    mut v218: D::F32Vec,
    mut v219: D::F32Vec,
    mut v220: D::F32Vec,
    mut v221: D::F32Vec,
    mut v222: D::F32Vec,
    mut v223: D::F32Vec,
    mut v224: D::F32Vec,
    mut v225: D::F32Vec,
    mut v226: D::F32Vec,
    mut v227: D::F32Vec,
    mut v228: D::F32Vec,
    mut v229: D::F32Vec,
    mut v230: D::F32Vec,
    mut v231: D::F32Vec,
    mut v232: D::F32Vec,
    mut v233: D::F32Vec,
    mut v234: D::F32Vec,
    mut v235: D::F32Vec,
    mut v236: D::F32Vec,
    mut v237: D::F32Vec,
    mut v238: D::F32Vec,
    mut v239: D::F32Vec,
    mut v240: D::F32Vec,
    mut v241: D::F32Vec,
    mut v242: D::F32Vec,
    mut v243: D::F32Vec,
    mut v244: D::F32Vec,
    mut v245: D::F32Vec,
    mut v246: D::F32Vec,
    mut v247: D::F32Vec,
    mut v248: D::F32Vec,
    mut v249: D::F32Vec,
    mut v250: D::F32Vec,
    mut v251: D::F32Vec,
    mut v252: D::F32Vec,
    mut v253: D::F32Vec,
    mut v254: D::F32Vec,
    mut v255: D::F32Vec,
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
                v126, v128, v130, v132, v134, v136, v138, v140, v142, v144, v146, v148, v150, v152,
                v154, v156, v158, v160, v162, v164, v166, v168, v170, v172, v174, v176, v178, v180,
                v182, v184, v186, v188, v190, v192, v194, v196, v198, v200, v202, v204, v206, v208,
                v210, v212, v214, v216, v218, v220, v222, v224, v226, v228, v230, v232, v234, v236,
                v238, v240, v242, v244, v246, v248, v250, v252, v254,
            ) = idct_128(
                d, v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v32,
                v34, v36, v38, v40, v42, v44, v46, v48, v50, v52, v54, v56, v58, v60, v62, v64,
                v66, v68, v70, v72, v74, v76, v78, v80, v82, v84, v86, v88, v90, v92, v94, v96,
                v98, v100, v102, v104, v106, v108, v110, v112, v114, v116, v118, v120, v122, v124,
                v126, v128, v130, v132, v134, v136, v138, v140, v142, v144, v146, v148, v150, v152,
                v154, v156, v158, v160, v162, v164, v166, v168, v170, v172, v174, v176, v178, v180,
                v182, v184, v186, v188, v190, v192, v194, v196, v198, v200, v202, v204, v206, v208,
                v210, v212, v214, v216, v218, v220, v222, v224, v226, v228, v230, v232, v234, v236,
                v238, v240, v242, v244, v246, v248, v250, v252, v254,
            );
        },
    );
    let mut v256 = v1 + v3;
    let mut v257 = v3 + v5;
    let mut v258 = v5 + v7;
    let mut v259 = v7 + v9;
    let mut v260 = v9 + v11;
    let mut v261 = v11 + v13;
    let mut v262 = v13 + v15;
    let mut v263 = v15 + v17;
    let mut v264 = v17 + v19;
    let mut v265 = v19 + v21;
    let mut v266 = v21 + v23;
    let mut v267 = v23 + v25;
    let mut v268 = v25 + v27;
    let mut v269 = v27 + v29;
    let mut v270 = v29 + v31;
    let mut v271 = v31 + v33;
    let mut v272 = v33 + v35;
    let mut v273 = v35 + v37;
    let mut v274 = v37 + v39;
    let mut v275 = v39 + v41;
    let mut v276 = v41 + v43;
    let mut v277 = v43 + v45;
    let mut v278 = v45 + v47;
    let mut v279 = v47 + v49;
    let mut v280 = v49 + v51;
    let mut v281 = v51 + v53;
    let mut v282 = v53 + v55;
    let mut v283 = v55 + v57;
    let mut v284 = v57 + v59;
    let mut v285 = v59 + v61;
    let mut v286 = v61 + v63;
    let mut v287 = v63 + v65;
    let mut v288 = v65 + v67;
    let mut v289 = v67 + v69;
    let mut v290 = v69 + v71;
    let mut v291 = v71 + v73;
    let mut v292 = v73 + v75;
    let mut v293 = v75 + v77;
    let mut v294 = v77 + v79;
    let mut v295 = v79 + v81;
    let mut v296 = v81 + v83;
    let mut v297 = v83 + v85;
    let mut v298 = v85 + v87;
    let mut v299 = v87 + v89;
    let mut v300 = v89 + v91;
    let mut v301 = v91 + v93;
    let mut v302 = v93 + v95;
    let mut v303 = v95 + v97;
    let mut v304 = v97 + v99;
    let mut v305 = v99 + v101;
    let mut v306 = v101 + v103;
    let mut v307 = v103 + v105;
    let mut v308 = v105 + v107;
    let mut v309 = v107 + v109;
    let mut v310 = v109 + v111;
    let mut v311 = v111 + v113;
    let mut v312 = v113 + v115;
    let mut v313 = v115 + v117;
    let mut v314 = v117 + v119;
    let mut v315 = v119 + v121;
    let mut v316 = v121 + v123;
    let mut v317 = v123 + v125;
    let mut v318 = v125 + v127;
    let mut v319 = v127 + v129;
    let mut v320 = v129 + v131;
    let mut v321 = v131 + v133;
    let mut v322 = v133 + v135;
    let mut v323 = v135 + v137;
    let mut v324 = v137 + v139;
    let mut v325 = v139 + v141;
    let mut v326 = v141 + v143;
    let mut v327 = v143 + v145;
    let mut v328 = v145 + v147;
    let mut v329 = v147 + v149;
    let mut v330 = v149 + v151;
    let mut v331 = v151 + v153;
    let mut v332 = v153 + v155;
    let mut v333 = v155 + v157;
    let mut v334 = v157 + v159;
    let mut v335 = v159 + v161;
    let mut v336 = v161 + v163;
    let mut v337 = v163 + v165;
    let mut v338 = v165 + v167;
    let mut v339 = v167 + v169;
    let mut v340 = v169 + v171;
    let mut v341 = v171 + v173;
    let mut v342 = v173 + v175;
    let mut v343 = v175 + v177;
    let mut v344 = v177 + v179;
    let mut v345 = v179 + v181;
    let mut v346 = v181 + v183;
    let mut v347 = v183 + v185;
    let mut v348 = v185 + v187;
    let mut v349 = v187 + v189;
    let mut v350 = v189 + v191;
    let mut v351 = v191 + v193;
    let mut v352 = v193 + v195;
    let mut v353 = v195 + v197;
    let mut v354 = v197 + v199;
    let mut v355 = v199 + v201;
    let mut v356 = v201 + v203;
    let mut v357 = v203 + v205;
    let mut v358 = v205 + v207;
    let mut v359 = v207 + v209;
    let mut v360 = v209 + v211;
    let mut v361 = v211 + v213;
    let mut v362 = v213 + v215;
    let mut v363 = v215 + v217;
    let mut v364 = v217 + v219;
    let mut v365 = v219 + v221;
    let mut v366 = v221 + v223;
    let mut v367 = v223 + v225;
    let mut v368 = v225 + v227;
    let mut v369 = v227 + v229;
    let mut v370 = v229 + v231;
    let mut v371 = v231 + v233;
    let mut v372 = v233 + v235;
    let mut v373 = v235 + v237;
    let mut v374 = v237 + v239;
    let mut v375 = v239 + v241;
    let mut v376 = v241 + v243;
    let mut v377 = v243 + v245;
    let mut v378 = v245 + v247;
    let mut v379 = v247 + v249;
    let mut v380 = v249 + v251;
    let mut v381 = v251 + v253;
    let mut v382 = v253 + v255;
    let mut v383 = v1 * D::F32Vec::splat(d, std::f32::consts::SQRT_2);
    d.call(
        #[inline(always)]
        |_| {
            (
                v383, v256, v257, v258, v259, v260, v261, v262, v263, v264, v265, v266, v267, v268,
                v269, v270, v271, v272, v273, v274, v275, v276, v277, v278, v279, v280, v281, v282,
                v283, v284, v285, v286, v287, v288, v289, v290, v291, v292, v293, v294, v295, v296,
                v297, v298, v299, v300, v301, v302, v303, v304, v305, v306, v307, v308, v309, v310,
                v311, v312, v313, v314, v315, v316, v317, v318, v319, v320, v321, v322, v323, v324,
                v325, v326, v327, v328, v329, v330, v331, v332, v333, v334, v335, v336, v337, v338,
                v339, v340, v341, v342, v343, v344, v345, v346, v347, v348, v349, v350, v351, v352,
                v353, v354, v355, v356, v357, v358, v359, v360, v361, v362, v363, v364, v365, v366,
                v367, v368, v369, v370, v371, v372, v373, v374, v375, v376, v377, v378, v379, v380,
                v381, v382,
            ) = idct_128(
                d, v383, v256, v257, v258, v259, v260, v261, v262, v263, v264, v265, v266, v267,
                v268, v269, v270, v271, v272, v273, v274, v275, v276, v277, v278, v279, v280, v281,
                v282, v283, v284, v285, v286, v287, v288, v289, v290, v291, v292, v293, v294, v295,
                v296, v297, v298, v299, v300, v301, v302, v303, v304, v305, v306, v307, v308, v309,
                v310, v311, v312, v313, v314, v315, v316, v317, v318, v319, v320, v321, v322, v323,
                v324, v325, v326, v327, v328, v329, v330, v331, v332, v333, v334, v335, v336, v337,
                v338, v339, v340, v341, v342, v343, v344, v345, v346, v347, v348, v349, v350, v351,
                v352, v353, v354, v355, v356, v357, v358, v359, v360, v361, v362, v363, v364, v365,
                v366, v367, v368, v369, v370, v371, v372, v373, v374, v375, v376, v377, v378, v379,
                v380, v381, v382,
            );
        },
    );
    let mul = D::F32Vec::splat(d, 0.5000094125358878);
    let mut v384 = v383.mul_add(mul, v0);
    let mut v385 = v383.neg_mul_add(mul, v0);
    let mul = D::F32Vec::splat(d, 0.5000847234557840);
    let mut v386 = v256.mul_add(mul, v2);
    let mut v387 = v256.neg_mul_add(mul, v2);
    let mul = D::F32Vec::splat(d, 0.5002354020255269);
    let mut v388 = v257.mul_add(mul, v4);
    let mut v389 = v257.neg_mul_add(mul, v4);
    let mul = D::F32Vec::splat(d, 0.5004615618093246);
    let mut v390 = v258.mul_add(mul, v6);
    let mut v391 = v258.neg_mul_add(mul, v6);
    let mul = D::F32Vec::splat(d, 0.5007633734146156);
    let mut v392 = v259.mul_add(mul, v8);
    let mut v393 = v259.neg_mul_add(mul, v8);
    let mul = D::F32Vec::splat(d, 0.5011410648064231);
    let mut v394 = v260.mul_add(mul, v10);
    let mut v395 = v260.neg_mul_add(mul, v10);
    let mul = D::F32Vec::splat(d, 0.5015949217281668);
    let mut v396 = v261.mul_add(mul, v12);
    let mut v397 = v261.neg_mul_add(mul, v12);
    let mul = D::F32Vec::splat(d, 0.5021252882303860);
    let mut v398 = v262.mul_add(mul, v14);
    let mut v399 = v262.neg_mul_add(mul, v14);
    let mul = D::F32Vec::splat(d, 0.5027325673091954);
    let mut v400 = v263.mul_add(mul, v16);
    let mut v401 = v263.neg_mul_add(mul, v16);
    let mul = D::F32Vec::splat(d, 0.5034172216566842);
    let mut v402 = v264.mul_add(mul, v18);
    let mut v403 = v264.neg_mul_add(mul, v18);
    let mul = D::F32Vec::splat(d, 0.5041797745258774);
    let mut v404 = v265.mul_add(mul, v20);
    let mut v405 = v265.neg_mul_add(mul, v20);
    let mul = D::F32Vec::splat(d, 0.5050208107132756);
    let mut v406 = v266.mul_add(mul, v22);
    let mut v407 = v266.neg_mul_add(mul, v22);
    let mul = D::F32Vec::splat(d, 0.5059409776624396);
    let mut v408 = v267.mul_add(mul, v24);
    let mut v409 = v267.neg_mul_add(mul, v24);
    let mul = D::F32Vec::splat(d, 0.5069409866925212);
    let mut v410 = v268.mul_add(mul, v26);
    let mut v411 = v268.neg_mul_add(mul, v26);
    let mul = D::F32Vec::splat(d, 0.5080216143561264);
    let mut v412 = v269.mul_add(mul, v28);
    let mut v413 = v269.neg_mul_add(mul, v28);
    let mul = D::F32Vec::splat(d, 0.5091837039313880);
    let mut v414 = v270.mul_add(mul, v30);
    let mut v415 = v270.neg_mul_add(mul, v30);
    let mul = D::F32Vec::splat(d, 0.5104281670536573);
    let mut v416 = v271.mul_add(mul, v32);
    let mut v417 = v271.neg_mul_add(mul, v32);
    let mul = D::F32Vec::splat(d, 0.5117559854927805);
    let mut v418 = v272.mul_add(mul, v34);
    let mut v419 = v272.neg_mul_add(mul, v34);
    let mul = D::F32Vec::splat(d, 0.5131682130825206);
    let mut v420 = v273.mul_add(mul, v36);
    let mut v421 = v273.neg_mul_add(mul, v36);
    let mul = D::F32Vec::splat(d, 0.5146659778093218);
    let mut v422 = v274.mul_add(mul, v38);
    let mut v423 = v274.neg_mul_add(mul, v38);
    let mul = D::F32Vec::splat(d, 0.5162504840682880);
    let mut v424 = v275.mul_add(mul, v40);
    let mut v425 = v275.neg_mul_add(mul, v40);
    let mul = D::F32Vec::splat(d, 0.5179230150949777);
    let mut v426 = v276.mul_add(mul, v42);
    let mut v427 = v276.neg_mul_add(mul, v42);
    let mul = D::F32Vec::splat(d, 0.5196849355823947);
    let mut v428 = v277.mul_add(mul, v44);
    let mut v429 = v277.neg_mul_add(mul, v44);
    let mul = D::F32Vec::splat(d, 0.5215376944933958);
    let mut v430 = v278.mul_add(mul, v46);
    let mut v431 = v278.neg_mul_add(mul, v46);
    let mul = D::F32Vec::splat(d, 0.5234828280796439);
    let mut v432 = v279.mul_add(mul, v48);
    let mut v433 = v279.neg_mul_add(mul, v48);
    let mul = D::F32Vec::splat(d, 0.5255219631192100);
    let mut v434 = v280.mul_add(mul, v50);
    let mut v435 = v280.neg_mul_add(mul, v50);
    let mul = D::F32Vec::splat(d, 0.5276568203859896);
    let mut v436 = v281.mul_add(mul, v52);
    let mut v437 = v281.neg_mul_add(mul, v52);
    let mul = D::F32Vec::splat(d, 0.5298892183652453);
    let mut v438 = v282.mul_add(mul, v54);
    let mut v439 = v282.neg_mul_add(mul, v54);
    let mul = D::F32Vec::splat(d, 0.5322210772308335);
    let mut v440 = v283.mul_add(mul, v56);
    let mut v441 = v283.neg_mul_add(mul, v56);
    let mul = D::F32Vec::splat(d, 0.5346544231010253);
    let mut v442 = v284.mul_add(mul, v58);
    let mut v443 = v284.neg_mul_add(mul, v58);
    let mul = D::F32Vec::splat(d, 0.5371913925913090);
    let mut v444 = v285.mul_add(mul, v60);
    let mut v445 = v285.neg_mul_add(mul, v60);
    let mul = D::F32Vec::splat(d, 0.5398342376841637);
    let mut v446 = v286.mul_add(mul, v62);
    let mut v447 = v286.neg_mul_add(mul, v62);
    let mul = D::F32Vec::splat(d, 0.5425853309375497);
    let mut v448 = v287.mul_add(mul, v64);
    let mut v449 = v287.neg_mul_add(mul, v64);
    let mul = D::F32Vec::splat(d, 0.5454471710557750);
    let mut v450 = v288.mul_add(mul, v66);
    let mut v451 = v288.neg_mul_add(mul, v66);
    let mul = D::F32Vec::splat(d, 0.5484223888484947);
    let mut v452 = v289.mul_add(mul, v68);
    let mut v453 = v289.neg_mul_add(mul, v68);
    let mul = D::F32Vec::splat(d, 0.5515137536058931);
    let mut v454 = v290.mul_add(mul, v70);
    let mut v455 = v290.neg_mul_add(mul, v70);
    let mul = D::F32Vec::splat(d, 0.5547241799206190);
    let mut v456 = v291.mul_add(mul, v72);
    let mut v457 = v291.neg_mul_add(mul, v72);
    let mul = D::F32Vec::splat(d, 0.5580567349898085);
    let mut v458 = v292.mul_add(mul, v74);
    let mut v459 = v292.neg_mul_add(mul, v74);
    let mul = D::F32Vec::splat(d, 0.5615146464335654);
    let mut v460 = v293.mul_add(mul, v76);
    let mut v461 = v293.neg_mul_add(mul, v76);
    let mul = D::F32Vec::splat(d, 0.5651013106696203);
    let mut v462 = v294.mul_add(mul, v78);
    let mut v463 = v294.neg_mul_add(mul, v78);
    let mul = D::F32Vec::splat(d, 0.5688203018875696);
    let mut v464 = v295.mul_add(mul, v80);
    let mut v465 = v295.neg_mul_add(mul, v80);
    let mul = D::F32Vec::splat(d, 0.5726753816701664);
    let mut v466 = v296.mul_add(mul, v82);
    let mut v467 = v296.neg_mul_add(mul, v82);
    let mul = D::F32Vec::splat(d, 0.5766705093136241);
    let mut v468 = v297.mul_add(mul, v84);
    let mut v469 = v297.neg_mul_add(mul, v84);
    let mul = D::F32Vec::splat(d, 0.5808098529038624);
    let mut v470 = v298.mul_add(mul, v86);
    let mut v471 = v298.neg_mul_add(mul, v86);
    let mul = D::F32Vec::splat(d, 0.5850978012111273);
    let mut v472 = v299.mul_add(mul, v88);
    let mut v473 = v299.neg_mul_add(mul, v88);
    let mul = D::F32Vec::splat(d, 0.5895389764715100);
    let mut v474 = v300.mul_add(mul, v90);
    let mut v475 = v300.neg_mul_add(mul, v90);
    let mul = D::F32Vec::splat(d, 0.5941382481306648);
    let mut v476 = v301.mul_add(mul, v92);
    let mut v477 = v301.neg_mul_add(mul, v92);
    let mul = D::F32Vec::splat(d, 0.5989007476325463);
    let mut v478 = v302.mul_add(mul, v94);
    let mut v479 = v302.neg_mul_add(mul, v94);
    let mul = D::F32Vec::splat(d, 0.6038318843443582);
    let mut v480 = v303.mul_add(mul, v96);
    let mut v481 = v303.neg_mul_add(mul, v96);
    let mul = D::F32Vec::splat(d, 0.6089373627182432);
    let mut v482 = v304.mul_add(mul, v98);
    let mut v483 = v304.neg_mul_add(mul, v98);
    let mul = D::F32Vec::splat(d, 0.6142232008006490);
    let mut v484 = v305.mul_add(mul, v100);
    let mut v485 = v305.neg_mul_add(mul, v100);
    let mul = D::F32Vec::splat(d, 0.6196957502119484);
    let mut v486 = v306.mul_add(mul, v102);
    let mut v487 = v306.neg_mul_add(mul, v102);
    let mul = D::F32Vec::splat(d, 0.6253617177319102);
    let mut v488 = v307.mul_add(mul, v104);
    let mut v489 = v307.neg_mul_add(mul, v104);
    let mul = D::F32Vec::splat(d, 0.6312281886412079);
    let mut v490 = v308.mul_add(mul, v106);
    let mut v491 = v308.neg_mul_add(mul, v106);
    let mul = D::F32Vec::splat(d, 0.6373026519855411);
    let mut v492 = v309.mul_add(mul, v108);
    let mut v493 = v309.neg_mul_add(mul, v108);
    let mul = D::F32Vec::splat(d, 0.6435930279473415);
    let mut v494 = v310.mul_add(mul, v110);
    let mut v495 = v310.neg_mul_add(mul, v110);
    let mul = D::F32Vec::splat(d, 0.6501076975307724);
    let mut v496 = v311.mul_add(mul, v112);
    let mut v497 = v311.neg_mul_add(mul, v112);
    let mul = D::F32Vec::splat(d, 0.6568555347890955);
    let mut v498 = v312.mul_add(mul, v114);
    let mut v499 = v312.neg_mul_add(mul, v114);
    let mul = D::F32Vec::splat(d, 0.6638459418498757);
    let mut v500 = v313.mul_add(mul, v116);
    let mut v501 = v313.neg_mul_add(mul, v116);
    let mul = D::F32Vec::splat(d, 0.6710888870233562);
    let mut v502 = v314.mul_add(mul, v118);
    let mut v503 = v314.neg_mul_add(mul, v118);
    let mul = D::F32Vec::splat(d, 0.6785949463131795);
    let mut v504 = v315.mul_add(mul, v120);
    let mut v505 = v315.neg_mul_add(mul, v120);
    let mul = D::F32Vec::splat(d, 0.6863753486870501);
    let mut v506 = v316.mul_add(mul, v122);
    let mut v507 = v316.neg_mul_add(mul, v122);
    let mul = D::F32Vec::splat(d, 0.6944420255086364);
    let mut v508 = v317.mul_add(mul, v124);
    let mut v509 = v317.neg_mul_add(mul, v124);
    let mul = D::F32Vec::splat(d, 0.7028076645818034);
    let mut v510 = v318.mul_add(mul, v126);
    let mut v511 = v318.neg_mul_add(mul, v126);
    let mul = D::F32Vec::splat(d, 0.7114857693151208);
    let mut v512 = v319.mul_add(mul, v128);
    let mut v513 = v319.neg_mul_add(mul, v128);
    let mul = D::F32Vec::splat(d, 0.7204907235796304);
    let mut v514 = v320.mul_add(mul, v130);
    let mut v515 = v320.neg_mul_add(mul, v130);
    let mul = D::F32Vec::splat(d, 0.7298378629074134);
    let mut v516 = v321.mul_add(mul, v132);
    let mut v517 = v321.neg_mul_add(mul, v132);
    let mul = D::F32Vec::splat(d, 0.7395435527641373);
    let mut v518 = v322.mul_add(mul, v134);
    let mut v519 = v322.neg_mul_add(mul, v134);
    let mul = D::F32Vec::splat(d, 0.7496252747273719);
    let mut v520 = v323.mul_add(mul, v136);
    let mut v521 = v323.neg_mul_add(mul, v136);
    let mul = D::F32Vec::splat(d, 0.7601017215162176);
    let mut v522 = v324.mul_add(mul, v138);
    let mut v523 = v324.neg_mul_add(mul, v138);
    let mul = D::F32Vec::splat(d, 0.7709929019493761);
    let mut v524 = v325.mul_add(mul, v140);
    let mut v525 = v325.neg_mul_add(mul, v140);
    let mul = D::F32Vec::splat(d, 0.7823202570613161);
    let mut v526 = v326.mul_add(mul, v142);
    let mut v527 = v326.neg_mul_add(mul, v142);
    let mul = D::F32Vec::splat(d, 0.7941067887834509);
    let mut v528 = v327.mul_add(mul, v144);
    let mut v529 = v327.neg_mul_add(mul, v144);
    let mul = D::F32Vec::splat(d, 0.8063772028037925);
    let mut v530 = v328.mul_add(mul, v146);
    let mut v531 = v328.neg_mul_add(mul, v146);
    let mul = D::F32Vec::splat(d, 0.8191580674598145);
    let mut v532 = v329.mul_add(mul, v148);
    let mut v533 = v329.neg_mul_add(mul, v148);
    let mul = D::F32Vec::splat(d, 0.8324779908019100);
    let mut v534 = v330.mul_add(mul, v150);
    let mut v535 = v330.neg_mul_add(mul, v150);
    let mul = D::F32Vec::splat(d, 0.8463678182968619);
    let mut v536 = v331.mul_add(mul, v152);
    let mut v537 = v331.neg_mul_add(mul, v152);
    let mul = D::F32Vec::splat(d, 0.8608608540319550);
    let mut v538 = v332.mul_add(mul, v154);
    let mut v539 = v332.neg_mul_add(mul, v154);
    let mul = D::F32Vec::splat(d, 0.8759931087426972);
    let mut v540 = v333.mul_add(mul, v156);
    let mut v541 = v333.neg_mul_add(mul, v156);
    let mul = D::F32Vec::splat(d, 0.8918035785352535);
    let mut v542 = v334.mul_add(mul, v158);
    let mut v543 = v334.neg_mul_add(mul, v158);
    let mul = D::F32Vec::splat(d, 0.9083345588266809);
    let mut v544 = v335.mul_add(mul, v160);
    let mut v545 = v335.neg_mul_add(mul, v160);
    let mul = D::F32Vec::splat(d, 0.9256319988042384);
    let mut v546 = v336.mul_add(mul, v162);
    let mut v547 = v336.neg_mul_add(mul, v162);
    let mul = D::F32Vec::splat(d, 0.9437459026371479);
    let mut v548 = v337.mul_add(mul, v164);
    let mut v549 = v337.neg_mul_add(mul, v164);
    let mul = D::F32Vec::splat(d, 0.9627307847948030);
    let mut v550 = v338.mul_add(mul, v166);
    let mut v551 = v338.neg_mul_add(mul, v166);
    let mul = D::F32Vec::splat(d, 0.9826461881778968);
    let mut v552 = v339.mul_add(mul, v168);
    let mut v553 = v339.neg_mul_add(mul, v168);
    let mul = D::F32Vec::splat(d, 1.0035572754078206);
    let mut v554 = v340.mul_add(mul, v170);
    let mut v555 = v340.neg_mul_add(mul, v170);
    let mul = D::F32Vec::splat(d, 1.0255355056139732);
    let mut v556 = v341.mul_add(mul, v172);
    let mut v557 = v341.neg_mul_add(mul, v172);
    let mul = D::F32Vec::splat(d, 1.0486594114961061);
    let mut v558 = v342.mul_add(mul, v174);
    let mut v559 = v342.neg_mul_add(mul, v174);
    let mul = D::F32Vec::splat(d, 1.0730154944316674);
    let mut v560 = v343.mul_add(mul, v176);
    let mut v561 = v343.neg_mul_add(mul, v176);
    let mul = D::F32Vec::splat(d, 1.0986992590905857);
    let mut v562 = v344.mul_add(mul, v178);
    let mut v563 = v344.neg_mul_add(mul, v178);
    let mul = D::F32Vec::splat(d, 1.1258164135986009);
    let mut v564 = v345.mul_add(mul, v180);
    let mut v565 = v345.neg_mul_add(mul, v180);
    let mul = D::F32Vec::splat(d, 1.1544842669978943);
    let mut v566 = v346.mul_add(mul, v182);
    let mut v567 = v346.neg_mul_add(mul, v182);
    let mul = D::F32Vec::splat(d, 1.1848333629084420);
    let mut v568 = v347.mul_add(mul, v184);
    let mut v569 = v347.neg_mul_add(mul, v184);
    let mul = D::F32Vec::splat(d, 1.2170093973146030);
    let mut v570 = v348.mul_add(mul, v186);
    let mut v571 = v348.neg_mul_add(mul, v186);
    let mul = D::F32Vec::splat(d, 1.2511754798461228);
    let mut v572 = v349.mul_add(mul, v188);
    let mut v573 = v349.neg_mul_add(mul, v188);
    let mul = D::F32Vec::splat(d, 1.2875148125367120);
    let mut v574 = v350.mul_add(mul, v190);
    let mut v575 = v350.neg_mul_add(mul, v190);
    let mul = D::F32Vec::splat(d, 1.3262338788327230);
    let mut v576 = v351.mul_add(mul, v192);
    let mut v577 = v351.neg_mul_add(mul, v192);
    let mul = D::F32Vec::splat(d, 1.3675662599582539);
    let mut v578 = v352.mul_add(mul, v194);
    let mut v579 = v352.neg_mul_add(mul, v194);
    let mul = D::F32Vec::splat(d, 1.4117772275006610);
    let mut v580 = v353.mul_add(mul, v196);
    let mut v581 = v353.neg_mul_add(mul, v196);
    let mul = D::F32Vec::splat(d, 1.4591693028668571);
    let mut v582 = v354.mul_add(mul, v198);
    let mut v583 = v354.neg_mul_add(mul, v198);
    let mul = D::F32Vec::splat(d, 1.5100890297227016);
    let mut v584 = v355.mul_add(mul, v200);
    let mut v585 = v355.neg_mul_add(mul, v200);
    let mul = D::F32Vec::splat(d, 1.5649352798258847);
    let mut v586 = v356.mul_add(mul, v202);
    let mut v587 = v356.neg_mul_add(mul, v202);
    let mul = D::F32Vec::splat(d, 1.6241695131835794);
    let mut v588 = v357.mul_add(mul, v204);
    let mut v589 = v357.neg_mul_add(mul, v204);
    let mul = D::F32Vec::splat(d, 1.6883285509131505);
    let mut v590 = v358.mul_add(mul, v206);
    let mut v591 = v358.neg_mul_add(mul, v206);
    let mul = D::F32Vec::splat(d, 1.7580406092704062);
    let mut v592 = v359.mul_add(mul, v208);
    let mut v593 = v359.neg_mul_add(mul, v208);
    let mul = D::F32Vec::splat(d, 1.8340456094306077);
    let mut v594 = v360.mul_add(mul, v210);
    let mut v595 = v360.neg_mul_add(mul, v210);
    let mul = D::F32Vec::splat(d, 1.9172211551275689);
    let mut v596 = v361.mul_add(mul, v212);
    let mut v597 = v361.neg_mul_add(mul, v212);
    let mul = D::F32Vec::splat(d, 2.0086161135167564);
    let mut v598 = v362.mul_add(mul, v214);
    let mut v599 = v362.neg_mul_add(mul, v214);
    let mul = D::F32Vec::splat(d, 2.1094945286246385);
    let mut v600 = v363.mul_add(mul, v216);
    let mut v601 = v363.neg_mul_add(mul, v216);
    let mul = D::F32Vec::splat(d, 2.2213937770112699);
    let mut v602 = v364.mul_add(mul, v218);
    let mut v603 = v364.neg_mul_add(mul, v218);
    let mul = D::F32Vec::splat(d, 2.3462026625311561);
    let mut v604 = v365.mul_add(mul, v220);
    let mut v605 = v365.neg_mul_add(mul, v220);
    let mul = D::F32Vec::splat(d, 2.4862679092035931);
    let mut v606 = v366.mul_add(mul, v222);
    let mut v607 = v366.neg_mul_add(mul, v222);
    let mul = D::F32Vec::splat(d, 2.6445418771448610);
    let mut v608 = v367.mul_add(mul, v224);
    let mut v609 = v367.neg_mul_add(mul, v224);
    let mul = D::F32Vec::splat(d, 2.8247914023505509);
    let mut v610 = v368.mul_add(mul, v226);
    let mut v611 = v368.neg_mul_add(mul, v226);
    let mul = D::F32Vec::splat(d, 3.0318994541759925);
    let mut v612 = v369.mul_add(mul, v228);
    let mut v613 = v369.neg_mul_add(mul, v228);
    let mul = D::F32Vec::splat(d, 3.2723115884254845);
    let mut v614 = v370.mul_add(mul, v230);
    let mut v615 = v370.neg_mul_add(mul, v230);
    let mul = D::F32Vec::splat(d, 3.5547153325075804);
    let mut v616 = v371.mul_add(mul, v232);
    let mut v617 = v371.neg_mul_add(mul, v232);
    let mul = D::F32Vec::splat(d, 3.8911077907003069);
    let mut v618 = v372.mul_add(mul, v234);
    let mut v619 = v372.neg_mul_add(mul, v234);
    let mul = D::F32Vec::splat(d, 4.2985375264490537);
    let mut v620 = v373.mul_add(mul, v236);
    let mut v621 = v373.neg_mul_add(mul, v236);
    let mul = D::F32Vec::splat(d, 4.8020760086650478);
    let mut v622 = v374.mul_add(mul, v238);
    let mut v623 = v374.neg_mul_add(mul, v238);
    let mul = D::F32Vec::splat(d, 5.4401662150913292);
    let mut v624 = v375.mul_add(mul, v240);
    let mut v625 = v375.neg_mul_add(mul, v240);
    let mul = D::F32Vec::splat(d, 6.2749084080393391);
    let mut v626 = v376.mul_add(mul, v242);
    let mut v627 = v376.neg_mul_add(mul, v242);
    let mul = D::F32Vec::splat(d, 7.4135667564223029);
    let mut v628 = v377.mul_add(mul, v244);
    let mut v629 = v377.neg_mul_add(mul, v244);
    let mul = D::F32Vec::splat(d, 9.0587514538797027);
    let mut v630 = v378.mul_add(mul, v246);
    let mut v631 = v378.neg_mul_add(mul, v246);
    let mul = D::F32Vec::splat(d, 11.6446273251750370);
    let mut v632 = v379.mul_add(mul, v248);
    let mut v633 = v379.neg_mul_add(mul, v248);
    let mul = D::F32Vec::splat(d, 16.3000230880315549);
    let mut v634 = v380.mul_add(mul, v250);
    let mut v635 = v380.neg_mul_add(mul, v250);
    let mul = D::F32Vec::splat(d, 27.1639776624482323);
    let mut v636 = v381.mul_add(mul, v252);
    let mut v637 = v381.neg_mul_add(mul, v252);
    let mul = D::F32Vec::splat(d, 81.4878421922251590);
    let mut v638 = v382.mul_add(mul, v254);
    let mut v639 = v382.neg_mul_add(mul, v254);
    (
        v384, v386, v388, v390, v392, v394, v396, v398, v400, v402, v404, v406, v408, v410, v412,
        v414, v416, v418, v420, v422, v424, v426, v428, v430, v432, v434, v436, v438, v440, v442,
        v444, v446, v448, v450, v452, v454, v456, v458, v460, v462, v464, v466, v468, v470, v472,
        v474, v476, v478, v480, v482, v484, v486, v488, v490, v492, v494, v496, v498, v500, v502,
        v504, v506, v508, v510, v512, v514, v516, v518, v520, v522, v524, v526, v528, v530, v532,
        v534, v536, v538, v540, v542, v544, v546, v548, v550, v552, v554, v556, v558, v560, v562,
        v564, v566, v568, v570, v572, v574, v576, v578, v580, v582, v584, v586, v588, v590, v592,
        v594, v596, v598, v600, v602, v604, v606, v608, v610, v612, v614, v616, v618, v620, v622,
        v624, v626, v628, v630, v632, v634, v636, v638, v639, v637, v635, v633, v631, v629, v627,
        v625, v623, v621, v619, v617, v615, v613, v611, v609, v607, v605, v603, v601, v599, v597,
        v595, v593, v591, v589, v587, v585, v583, v581, v579, v577, v575, v573, v571, v569, v567,
        v565, v563, v561, v559, v557, v555, v553, v551, v549, v547, v545, v543, v541, v539, v537,
        v535, v533, v531, v529, v527, v525, v523, v521, v519, v517, v515, v513, v511, v509, v507,
        v505, v503, v501, v499, v497, v495, v493, v491, v489, v487, v485, v483, v481, v479, v477,
        v475, v473, v471, v469, v467, v465, v463, v461, v459, v457, v455, v453, v451, v449, v447,
        v445, v443, v441, v439, v437, v435, v433, v431, v429, v427, v425, v423, v421, v419, v417,
        v415, v413, v411, v409, v407, v405, v403, v401, v399, v397, v395, v393, v391, v389, v387,
        v385,
    )
}

#[inline(always)]
pub(super) fn do_idct_256<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
    stride: usize,
) {
    assert!(data.len() > 255 * stride);
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
    let mut v128 = D::F32Vec::load_array(d, &data[128 * stride]);
    let mut v129 = D::F32Vec::load_array(d, &data[129 * stride]);
    let mut v130 = D::F32Vec::load_array(d, &data[130 * stride]);
    let mut v131 = D::F32Vec::load_array(d, &data[131 * stride]);
    let mut v132 = D::F32Vec::load_array(d, &data[132 * stride]);
    let mut v133 = D::F32Vec::load_array(d, &data[133 * stride]);
    let mut v134 = D::F32Vec::load_array(d, &data[134 * stride]);
    let mut v135 = D::F32Vec::load_array(d, &data[135 * stride]);
    let mut v136 = D::F32Vec::load_array(d, &data[136 * stride]);
    let mut v137 = D::F32Vec::load_array(d, &data[137 * stride]);
    let mut v138 = D::F32Vec::load_array(d, &data[138 * stride]);
    let mut v139 = D::F32Vec::load_array(d, &data[139 * stride]);
    let mut v140 = D::F32Vec::load_array(d, &data[140 * stride]);
    let mut v141 = D::F32Vec::load_array(d, &data[141 * stride]);
    let mut v142 = D::F32Vec::load_array(d, &data[142 * stride]);
    let mut v143 = D::F32Vec::load_array(d, &data[143 * stride]);
    let mut v144 = D::F32Vec::load_array(d, &data[144 * stride]);
    let mut v145 = D::F32Vec::load_array(d, &data[145 * stride]);
    let mut v146 = D::F32Vec::load_array(d, &data[146 * stride]);
    let mut v147 = D::F32Vec::load_array(d, &data[147 * stride]);
    let mut v148 = D::F32Vec::load_array(d, &data[148 * stride]);
    let mut v149 = D::F32Vec::load_array(d, &data[149 * stride]);
    let mut v150 = D::F32Vec::load_array(d, &data[150 * stride]);
    let mut v151 = D::F32Vec::load_array(d, &data[151 * stride]);
    let mut v152 = D::F32Vec::load_array(d, &data[152 * stride]);
    let mut v153 = D::F32Vec::load_array(d, &data[153 * stride]);
    let mut v154 = D::F32Vec::load_array(d, &data[154 * stride]);
    let mut v155 = D::F32Vec::load_array(d, &data[155 * stride]);
    let mut v156 = D::F32Vec::load_array(d, &data[156 * stride]);
    let mut v157 = D::F32Vec::load_array(d, &data[157 * stride]);
    let mut v158 = D::F32Vec::load_array(d, &data[158 * stride]);
    let mut v159 = D::F32Vec::load_array(d, &data[159 * stride]);
    let mut v160 = D::F32Vec::load_array(d, &data[160 * stride]);
    let mut v161 = D::F32Vec::load_array(d, &data[161 * stride]);
    let mut v162 = D::F32Vec::load_array(d, &data[162 * stride]);
    let mut v163 = D::F32Vec::load_array(d, &data[163 * stride]);
    let mut v164 = D::F32Vec::load_array(d, &data[164 * stride]);
    let mut v165 = D::F32Vec::load_array(d, &data[165 * stride]);
    let mut v166 = D::F32Vec::load_array(d, &data[166 * stride]);
    let mut v167 = D::F32Vec::load_array(d, &data[167 * stride]);
    let mut v168 = D::F32Vec::load_array(d, &data[168 * stride]);
    let mut v169 = D::F32Vec::load_array(d, &data[169 * stride]);
    let mut v170 = D::F32Vec::load_array(d, &data[170 * stride]);
    let mut v171 = D::F32Vec::load_array(d, &data[171 * stride]);
    let mut v172 = D::F32Vec::load_array(d, &data[172 * stride]);
    let mut v173 = D::F32Vec::load_array(d, &data[173 * stride]);
    let mut v174 = D::F32Vec::load_array(d, &data[174 * stride]);
    let mut v175 = D::F32Vec::load_array(d, &data[175 * stride]);
    let mut v176 = D::F32Vec::load_array(d, &data[176 * stride]);
    let mut v177 = D::F32Vec::load_array(d, &data[177 * stride]);
    let mut v178 = D::F32Vec::load_array(d, &data[178 * stride]);
    let mut v179 = D::F32Vec::load_array(d, &data[179 * stride]);
    let mut v180 = D::F32Vec::load_array(d, &data[180 * stride]);
    let mut v181 = D::F32Vec::load_array(d, &data[181 * stride]);
    let mut v182 = D::F32Vec::load_array(d, &data[182 * stride]);
    let mut v183 = D::F32Vec::load_array(d, &data[183 * stride]);
    let mut v184 = D::F32Vec::load_array(d, &data[184 * stride]);
    let mut v185 = D::F32Vec::load_array(d, &data[185 * stride]);
    let mut v186 = D::F32Vec::load_array(d, &data[186 * stride]);
    let mut v187 = D::F32Vec::load_array(d, &data[187 * stride]);
    let mut v188 = D::F32Vec::load_array(d, &data[188 * stride]);
    let mut v189 = D::F32Vec::load_array(d, &data[189 * stride]);
    let mut v190 = D::F32Vec::load_array(d, &data[190 * stride]);
    let mut v191 = D::F32Vec::load_array(d, &data[191 * stride]);
    let mut v192 = D::F32Vec::load_array(d, &data[192 * stride]);
    let mut v193 = D::F32Vec::load_array(d, &data[193 * stride]);
    let mut v194 = D::F32Vec::load_array(d, &data[194 * stride]);
    let mut v195 = D::F32Vec::load_array(d, &data[195 * stride]);
    let mut v196 = D::F32Vec::load_array(d, &data[196 * stride]);
    let mut v197 = D::F32Vec::load_array(d, &data[197 * stride]);
    let mut v198 = D::F32Vec::load_array(d, &data[198 * stride]);
    let mut v199 = D::F32Vec::load_array(d, &data[199 * stride]);
    let mut v200 = D::F32Vec::load_array(d, &data[200 * stride]);
    let mut v201 = D::F32Vec::load_array(d, &data[201 * stride]);
    let mut v202 = D::F32Vec::load_array(d, &data[202 * stride]);
    let mut v203 = D::F32Vec::load_array(d, &data[203 * stride]);
    let mut v204 = D::F32Vec::load_array(d, &data[204 * stride]);
    let mut v205 = D::F32Vec::load_array(d, &data[205 * stride]);
    let mut v206 = D::F32Vec::load_array(d, &data[206 * stride]);
    let mut v207 = D::F32Vec::load_array(d, &data[207 * stride]);
    let mut v208 = D::F32Vec::load_array(d, &data[208 * stride]);
    let mut v209 = D::F32Vec::load_array(d, &data[209 * stride]);
    let mut v210 = D::F32Vec::load_array(d, &data[210 * stride]);
    let mut v211 = D::F32Vec::load_array(d, &data[211 * stride]);
    let mut v212 = D::F32Vec::load_array(d, &data[212 * stride]);
    let mut v213 = D::F32Vec::load_array(d, &data[213 * stride]);
    let mut v214 = D::F32Vec::load_array(d, &data[214 * stride]);
    let mut v215 = D::F32Vec::load_array(d, &data[215 * stride]);
    let mut v216 = D::F32Vec::load_array(d, &data[216 * stride]);
    let mut v217 = D::F32Vec::load_array(d, &data[217 * stride]);
    let mut v218 = D::F32Vec::load_array(d, &data[218 * stride]);
    let mut v219 = D::F32Vec::load_array(d, &data[219 * stride]);
    let mut v220 = D::F32Vec::load_array(d, &data[220 * stride]);
    let mut v221 = D::F32Vec::load_array(d, &data[221 * stride]);
    let mut v222 = D::F32Vec::load_array(d, &data[222 * stride]);
    let mut v223 = D::F32Vec::load_array(d, &data[223 * stride]);
    let mut v224 = D::F32Vec::load_array(d, &data[224 * stride]);
    let mut v225 = D::F32Vec::load_array(d, &data[225 * stride]);
    let mut v226 = D::F32Vec::load_array(d, &data[226 * stride]);
    let mut v227 = D::F32Vec::load_array(d, &data[227 * stride]);
    let mut v228 = D::F32Vec::load_array(d, &data[228 * stride]);
    let mut v229 = D::F32Vec::load_array(d, &data[229 * stride]);
    let mut v230 = D::F32Vec::load_array(d, &data[230 * stride]);
    let mut v231 = D::F32Vec::load_array(d, &data[231 * stride]);
    let mut v232 = D::F32Vec::load_array(d, &data[232 * stride]);
    let mut v233 = D::F32Vec::load_array(d, &data[233 * stride]);
    let mut v234 = D::F32Vec::load_array(d, &data[234 * stride]);
    let mut v235 = D::F32Vec::load_array(d, &data[235 * stride]);
    let mut v236 = D::F32Vec::load_array(d, &data[236 * stride]);
    let mut v237 = D::F32Vec::load_array(d, &data[237 * stride]);
    let mut v238 = D::F32Vec::load_array(d, &data[238 * stride]);
    let mut v239 = D::F32Vec::load_array(d, &data[239 * stride]);
    let mut v240 = D::F32Vec::load_array(d, &data[240 * stride]);
    let mut v241 = D::F32Vec::load_array(d, &data[241 * stride]);
    let mut v242 = D::F32Vec::load_array(d, &data[242 * stride]);
    let mut v243 = D::F32Vec::load_array(d, &data[243 * stride]);
    let mut v244 = D::F32Vec::load_array(d, &data[244 * stride]);
    let mut v245 = D::F32Vec::load_array(d, &data[245 * stride]);
    let mut v246 = D::F32Vec::load_array(d, &data[246 * stride]);
    let mut v247 = D::F32Vec::load_array(d, &data[247 * stride]);
    let mut v248 = D::F32Vec::load_array(d, &data[248 * stride]);
    let mut v249 = D::F32Vec::load_array(d, &data[249 * stride]);
    let mut v250 = D::F32Vec::load_array(d, &data[250 * stride]);
    let mut v251 = D::F32Vec::load_array(d, &data[251 * stride]);
    let mut v252 = D::F32Vec::load_array(d, &data[252 * stride]);
    let mut v253 = D::F32Vec::load_array(d, &data[253 * stride]);
    let mut v254 = D::F32Vec::load_array(d, &data[254 * stride]);
    let mut v255 = D::F32Vec::load_array(d, &data[255 * stride]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73,
        v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91,
        v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107,
        v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122,
        v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137,
        v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152,
        v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166, v167,
        v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181, v182,
        v183, v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194, v195, v196, v197,
        v198, v199, v200, v201, v202, v203, v204, v205, v206, v207, v208, v209, v210, v211, v212,
        v213, v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227,
        v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241, v242,
        v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255,
    ) = idct_256(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72,
        v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90,
        v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106,
        v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121,
        v122, v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136,
        v137, v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151,
        v152, v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166,
        v167, v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181,
        v182, v183, v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194, v195, v196,
        v197, v198, v199, v200, v201, v202, v203, v204, v205, v206, v207, v208, v209, v210, v211,
        v212, v213, v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226,
        v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241,
        v242, v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255,
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
    v128.store_array(&mut data[128 * stride]);
    v129.store_array(&mut data[129 * stride]);
    v130.store_array(&mut data[130 * stride]);
    v131.store_array(&mut data[131 * stride]);
    v132.store_array(&mut data[132 * stride]);
    v133.store_array(&mut data[133 * stride]);
    v134.store_array(&mut data[134 * stride]);
    v135.store_array(&mut data[135 * stride]);
    v136.store_array(&mut data[136 * stride]);
    v137.store_array(&mut data[137 * stride]);
    v138.store_array(&mut data[138 * stride]);
    v139.store_array(&mut data[139 * stride]);
    v140.store_array(&mut data[140 * stride]);
    v141.store_array(&mut data[141 * stride]);
    v142.store_array(&mut data[142 * stride]);
    v143.store_array(&mut data[143 * stride]);
    v144.store_array(&mut data[144 * stride]);
    v145.store_array(&mut data[145 * stride]);
    v146.store_array(&mut data[146 * stride]);
    v147.store_array(&mut data[147 * stride]);
    v148.store_array(&mut data[148 * stride]);
    v149.store_array(&mut data[149 * stride]);
    v150.store_array(&mut data[150 * stride]);
    v151.store_array(&mut data[151 * stride]);
    v152.store_array(&mut data[152 * stride]);
    v153.store_array(&mut data[153 * stride]);
    v154.store_array(&mut data[154 * stride]);
    v155.store_array(&mut data[155 * stride]);
    v156.store_array(&mut data[156 * stride]);
    v157.store_array(&mut data[157 * stride]);
    v158.store_array(&mut data[158 * stride]);
    v159.store_array(&mut data[159 * stride]);
    v160.store_array(&mut data[160 * stride]);
    v161.store_array(&mut data[161 * stride]);
    v162.store_array(&mut data[162 * stride]);
    v163.store_array(&mut data[163 * stride]);
    v164.store_array(&mut data[164 * stride]);
    v165.store_array(&mut data[165 * stride]);
    v166.store_array(&mut data[166 * stride]);
    v167.store_array(&mut data[167 * stride]);
    v168.store_array(&mut data[168 * stride]);
    v169.store_array(&mut data[169 * stride]);
    v170.store_array(&mut data[170 * stride]);
    v171.store_array(&mut data[171 * stride]);
    v172.store_array(&mut data[172 * stride]);
    v173.store_array(&mut data[173 * stride]);
    v174.store_array(&mut data[174 * stride]);
    v175.store_array(&mut data[175 * stride]);
    v176.store_array(&mut data[176 * stride]);
    v177.store_array(&mut data[177 * stride]);
    v178.store_array(&mut data[178 * stride]);
    v179.store_array(&mut data[179 * stride]);
    v180.store_array(&mut data[180 * stride]);
    v181.store_array(&mut data[181 * stride]);
    v182.store_array(&mut data[182 * stride]);
    v183.store_array(&mut data[183 * stride]);
    v184.store_array(&mut data[184 * stride]);
    v185.store_array(&mut data[185 * stride]);
    v186.store_array(&mut data[186 * stride]);
    v187.store_array(&mut data[187 * stride]);
    v188.store_array(&mut data[188 * stride]);
    v189.store_array(&mut data[189 * stride]);
    v190.store_array(&mut data[190 * stride]);
    v191.store_array(&mut data[191 * stride]);
    v192.store_array(&mut data[192 * stride]);
    v193.store_array(&mut data[193 * stride]);
    v194.store_array(&mut data[194 * stride]);
    v195.store_array(&mut data[195 * stride]);
    v196.store_array(&mut data[196 * stride]);
    v197.store_array(&mut data[197 * stride]);
    v198.store_array(&mut data[198 * stride]);
    v199.store_array(&mut data[199 * stride]);
    v200.store_array(&mut data[200 * stride]);
    v201.store_array(&mut data[201 * stride]);
    v202.store_array(&mut data[202 * stride]);
    v203.store_array(&mut data[203 * stride]);
    v204.store_array(&mut data[204 * stride]);
    v205.store_array(&mut data[205 * stride]);
    v206.store_array(&mut data[206 * stride]);
    v207.store_array(&mut data[207 * stride]);
    v208.store_array(&mut data[208 * stride]);
    v209.store_array(&mut data[209 * stride]);
    v210.store_array(&mut data[210 * stride]);
    v211.store_array(&mut data[211 * stride]);
    v212.store_array(&mut data[212 * stride]);
    v213.store_array(&mut data[213 * stride]);
    v214.store_array(&mut data[214 * stride]);
    v215.store_array(&mut data[215 * stride]);
    v216.store_array(&mut data[216 * stride]);
    v217.store_array(&mut data[217 * stride]);
    v218.store_array(&mut data[218 * stride]);
    v219.store_array(&mut data[219 * stride]);
    v220.store_array(&mut data[220 * stride]);
    v221.store_array(&mut data[221 * stride]);
    v222.store_array(&mut data[222 * stride]);
    v223.store_array(&mut data[223 * stride]);
    v224.store_array(&mut data[224 * stride]);
    v225.store_array(&mut data[225 * stride]);
    v226.store_array(&mut data[226 * stride]);
    v227.store_array(&mut data[227 * stride]);
    v228.store_array(&mut data[228 * stride]);
    v229.store_array(&mut data[229 * stride]);
    v230.store_array(&mut data[230 * stride]);
    v231.store_array(&mut data[231 * stride]);
    v232.store_array(&mut data[232 * stride]);
    v233.store_array(&mut data[233 * stride]);
    v234.store_array(&mut data[234 * stride]);
    v235.store_array(&mut data[235 * stride]);
    v236.store_array(&mut data[236 * stride]);
    v237.store_array(&mut data[237 * stride]);
    v238.store_array(&mut data[238 * stride]);
    v239.store_array(&mut data[239 * stride]);
    v240.store_array(&mut data[240 * stride]);
    v241.store_array(&mut data[241 * stride]);
    v242.store_array(&mut data[242 * stride]);
    v243.store_array(&mut data[243 * stride]);
    v244.store_array(&mut data[244 * stride]);
    v245.store_array(&mut data[245 * stride]);
    v246.store_array(&mut data[246 * stride]);
    v247.store_array(&mut data[247 * stride]);
    v248.store_array(&mut data[248 * stride]);
    v249.store_array(&mut data[249 * stride]);
    v250.store_array(&mut data[250 * stride]);
    v251.store_array(&mut data[251 * stride]);
    v252.store_array(&mut data[252 * stride]);
    v253.store_array(&mut data[253 * stride]);
    v254.store_array(&mut data[254 * stride]);
    v255.store_array(&mut data[255 * stride]);
}

#[inline(always)]
pub(super) fn do_idct_256_rowblock<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
) {
    assert!(data.len() >= 256);
    const { assert!(256usize.is_multiple_of(D::F32Vec::LEN)) };
    let row_stride = 256 / D::F32Vec::LEN;
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
    let mut v128 = D::F32Vec::load_array(
        d,
        &data[row_stride * (128 % D::F32Vec::LEN) + (128 / D::F32Vec::LEN)],
    );
    let mut v129 = D::F32Vec::load_array(
        d,
        &data[row_stride * (129 % D::F32Vec::LEN) + (129 / D::F32Vec::LEN)],
    );
    let mut v130 = D::F32Vec::load_array(
        d,
        &data[row_stride * (130 % D::F32Vec::LEN) + (130 / D::F32Vec::LEN)],
    );
    let mut v131 = D::F32Vec::load_array(
        d,
        &data[row_stride * (131 % D::F32Vec::LEN) + (131 / D::F32Vec::LEN)],
    );
    let mut v132 = D::F32Vec::load_array(
        d,
        &data[row_stride * (132 % D::F32Vec::LEN) + (132 / D::F32Vec::LEN)],
    );
    let mut v133 = D::F32Vec::load_array(
        d,
        &data[row_stride * (133 % D::F32Vec::LEN) + (133 / D::F32Vec::LEN)],
    );
    let mut v134 = D::F32Vec::load_array(
        d,
        &data[row_stride * (134 % D::F32Vec::LEN) + (134 / D::F32Vec::LEN)],
    );
    let mut v135 = D::F32Vec::load_array(
        d,
        &data[row_stride * (135 % D::F32Vec::LEN) + (135 / D::F32Vec::LEN)],
    );
    let mut v136 = D::F32Vec::load_array(
        d,
        &data[row_stride * (136 % D::F32Vec::LEN) + (136 / D::F32Vec::LEN)],
    );
    let mut v137 = D::F32Vec::load_array(
        d,
        &data[row_stride * (137 % D::F32Vec::LEN) + (137 / D::F32Vec::LEN)],
    );
    let mut v138 = D::F32Vec::load_array(
        d,
        &data[row_stride * (138 % D::F32Vec::LEN) + (138 / D::F32Vec::LEN)],
    );
    let mut v139 = D::F32Vec::load_array(
        d,
        &data[row_stride * (139 % D::F32Vec::LEN) + (139 / D::F32Vec::LEN)],
    );
    let mut v140 = D::F32Vec::load_array(
        d,
        &data[row_stride * (140 % D::F32Vec::LEN) + (140 / D::F32Vec::LEN)],
    );
    let mut v141 = D::F32Vec::load_array(
        d,
        &data[row_stride * (141 % D::F32Vec::LEN) + (141 / D::F32Vec::LEN)],
    );
    let mut v142 = D::F32Vec::load_array(
        d,
        &data[row_stride * (142 % D::F32Vec::LEN) + (142 / D::F32Vec::LEN)],
    );
    let mut v143 = D::F32Vec::load_array(
        d,
        &data[row_stride * (143 % D::F32Vec::LEN) + (143 / D::F32Vec::LEN)],
    );
    let mut v144 = D::F32Vec::load_array(
        d,
        &data[row_stride * (144 % D::F32Vec::LEN) + (144 / D::F32Vec::LEN)],
    );
    let mut v145 = D::F32Vec::load_array(
        d,
        &data[row_stride * (145 % D::F32Vec::LEN) + (145 / D::F32Vec::LEN)],
    );
    let mut v146 = D::F32Vec::load_array(
        d,
        &data[row_stride * (146 % D::F32Vec::LEN) + (146 / D::F32Vec::LEN)],
    );
    let mut v147 = D::F32Vec::load_array(
        d,
        &data[row_stride * (147 % D::F32Vec::LEN) + (147 / D::F32Vec::LEN)],
    );
    let mut v148 = D::F32Vec::load_array(
        d,
        &data[row_stride * (148 % D::F32Vec::LEN) + (148 / D::F32Vec::LEN)],
    );
    let mut v149 = D::F32Vec::load_array(
        d,
        &data[row_stride * (149 % D::F32Vec::LEN) + (149 / D::F32Vec::LEN)],
    );
    let mut v150 = D::F32Vec::load_array(
        d,
        &data[row_stride * (150 % D::F32Vec::LEN) + (150 / D::F32Vec::LEN)],
    );
    let mut v151 = D::F32Vec::load_array(
        d,
        &data[row_stride * (151 % D::F32Vec::LEN) + (151 / D::F32Vec::LEN)],
    );
    let mut v152 = D::F32Vec::load_array(
        d,
        &data[row_stride * (152 % D::F32Vec::LEN) + (152 / D::F32Vec::LEN)],
    );
    let mut v153 = D::F32Vec::load_array(
        d,
        &data[row_stride * (153 % D::F32Vec::LEN) + (153 / D::F32Vec::LEN)],
    );
    let mut v154 = D::F32Vec::load_array(
        d,
        &data[row_stride * (154 % D::F32Vec::LEN) + (154 / D::F32Vec::LEN)],
    );
    let mut v155 = D::F32Vec::load_array(
        d,
        &data[row_stride * (155 % D::F32Vec::LEN) + (155 / D::F32Vec::LEN)],
    );
    let mut v156 = D::F32Vec::load_array(
        d,
        &data[row_stride * (156 % D::F32Vec::LEN) + (156 / D::F32Vec::LEN)],
    );
    let mut v157 = D::F32Vec::load_array(
        d,
        &data[row_stride * (157 % D::F32Vec::LEN) + (157 / D::F32Vec::LEN)],
    );
    let mut v158 = D::F32Vec::load_array(
        d,
        &data[row_stride * (158 % D::F32Vec::LEN) + (158 / D::F32Vec::LEN)],
    );
    let mut v159 = D::F32Vec::load_array(
        d,
        &data[row_stride * (159 % D::F32Vec::LEN) + (159 / D::F32Vec::LEN)],
    );
    let mut v160 = D::F32Vec::load_array(
        d,
        &data[row_stride * (160 % D::F32Vec::LEN) + (160 / D::F32Vec::LEN)],
    );
    let mut v161 = D::F32Vec::load_array(
        d,
        &data[row_stride * (161 % D::F32Vec::LEN) + (161 / D::F32Vec::LEN)],
    );
    let mut v162 = D::F32Vec::load_array(
        d,
        &data[row_stride * (162 % D::F32Vec::LEN) + (162 / D::F32Vec::LEN)],
    );
    let mut v163 = D::F32Vec::load_array(
        d,
        &data[row_stride * (163 % D::F32Vec::LEN) + (163 / D::F32Vec::LEN)],
    );
    let mut v164 = D::F32Vec::load_array(
        d,
        &data[row_stride * (164 % D::F32Vec::LEN) + (164 / D::F32Vec::LEN)],
    );
    let mut v165 = D::F32Vec::load_array(
        d,
        &data[row_stride * (165 % D::F32Vec::LEN) + (165 / D::F32Vec::LEN)],
    );
    let mut v166 = D::F32Vec::load_array(
        d,
        &data[row_stride * (166 % D::F32Vec::LEN) + (166 / D::F32Vec::LEN)],
    );
    let mut v167 = D::F32Vec::load_array(
        d,
        &data[row_stride * (167 % D::F32Vec::LEN) + (167 / D::F32Vec::LEN)],
    );
    let mut v168 = D::F32Vec::load_array(
        d,
        &data[row_stride * (168 % D::F32Vec::LEN) + (168 / D::F32Vec::LEN)],
    );
    let mut v169 = D::F32Vec::load_array(
        d,
        &data[row_stride * (169 % D::F32Vec::LEN) + (169 / D::F32Vec::LEN)],
    );
    let mut v170 = D::F32Vec::load_array(
        d,
        &data[row_stride * (170 % D::F32Vec::LEN) + (170 / D::F32Vec::LEN)],
    );
    let mut v171 = D::F32Vec::load_array(
        d,
        &data[row_stride * (171 % D::F32Vec::LEN) + (171 / D::F32Vec::LEN)],
    );
    let mut v172 = D::F32Vec::load_array(
        d,
        &data[row_stride * (172 % D::F32Vec::LEN) + (172 / D::F32Vec::LEN)],
    );
    let mut v173 = D::F32Vec::load_array(
        d,
        &data[row_stride * (173 % D::F32Vec::LEN) + (173 / D::F32Vec::LEN)],
    );
    let mut v174 = D::F32Vec::load_array(
        d,
        &data[row_stride * (174 % D::F32Vec::LEN) + (174 / D::F32Vec::LEN)],
    );
    let mut v175 = D::F32Vec::load_array(
        d,
        &data[row_stride * (175 % D::F32Vec::LEN) + (175 / D::F32Vec::LEN)],
    );
    let mut v176 = D::F32Vec::load_array(
        d,
        &data[row_stride * (176 % D::F32Vec::LEN) + (176 / D::F32Vec::LEN)],
    );
    let mut v177 = D::F32Vec::load_array(
        d,
        &data[row_stride * (177 % D::F32Vec::LEN) + (177 / D::F32Vec::LEN)],
    );
    let mut v178 = D::F32Vec::load_array(
        d,
        &data[row_stride * (178 % D::F32Vec::LEN) + (178 / D::F32Vec::LEN)],
    );
    let mut v179 = D::F32Vec::load_array(
        d,
        &data[row_stride * (179 % D::F32Vec::LEN) + (179 / D::F32Vec::LEN)],
    );
    let mut v180 = D::F32Vec::load_array(
        d,
        &data[row_stride * (180 % D::F32Vec::LEN) + (180 / D::F32Vec::LEN)],
    );
    let mut v181 = D::F32Vec::load_array(
        d,
        &data[row_stride * (181 % D::F32Vec::LEN) + (181 / D::F32Vec::LEN)],
    );
    let mut v182 = D::F32Vec::load_array(
        d,
        &data[row_stride * (182 % D::F32Vec::LEN) + (182 / D::F32Vec::LEN)],
    );
    let mut v183 = D::F32Vec::load_array(
        d,
        &data[row_stride * (183 % D::F32Vec::LEN) + (183 / D::F32Vec::LEN)],
    );
    let mut v184 = D::F32Vec::load_array(
        d,
        &data[row_stride * (184 % D::F32Vec::LEN) + (184 / D::F32Vec::LEN)],
    );
    let mut v185 = D::F32Vec::load_array(
        d,
        &data[row_stride * (185 % D::F32Vec::LEN) + (185 / D::F32Vec::LEN)],
    );
    let mut v186 = D::F32Vec::load_array(
        d,
        &data[row_stride * (186 % D::F32Vec::LEN) + (186 / D::F32Vec::LEN)],
    );
    let mut v187 = D::F32Vec::load_array(
        d,
        &data[row_stride * (187 % D::F32Vec::LEN) + (187 / D::F32Vec::LEN)],
    );
    let mut v188 = D::F32Vec::load_array(
        d,
        &data[row_stride * (188 % D::F32Vec::LEN) + (188 / D::F32Vec::LEN)],
    );
    let mut v189 = D::F32Vec::load_array(
        d,
        &data[row_stride * (189 % D::F32Vec::LEN) + (189 / D::F32Vec::LEN)],
    );
    let mut v190 = D::F32Vec::load_array(
        d,
        &data[row_stride * (190 % D::F32Vec::LEN) + (190 / D::F32Vec::LEN)],
    );
    let mut v191 = D::F32Vec::load_array(
        d,
        &data[row_stride * (191 % D::F32Vec::LEN) + (191 / D::F32Vec::LEN)],
    );
    let mut v192 = D::F32Vec::load_array(
        d,
        &data[row_stride * (192 % D::F32Vec::LEN) + (192 / D::F32Vec::LEN)],
    );
    let mut v193 = D::F32Vec::load_array(
        d,
        &data[row_stride * (193 % D::F32Vec::LEN) + (193 / D::F32Vec::LEN)],
    );
    let mut v194 = D::F32Vec::load_array(
        d,
        &data[row_stride * (194 % D::F32Vec::LEN) + (194 / D::F32Vec::LEN)],
    );
    let mut v195 = D::F32Vec::load_array(
        d,
        &data[row_stride * (195 % D::F32Vec::LEN) + (195 / D::F32Vec::LEN)],
    );
    let mut v196 = D::F32Vec::load_array(
        d,
        &data[row_stride * (196 % D::F32Vec::LEN) + (196 / D::F32Vec::LEN)],
    );
    let mut v197 = D::F32Vec::load_array(
        d,
        &data[row_stride * (197 % D::F32Vec::LEN) + (197 / D::F32Vec::LEN)],
    );
    let mut v198 = D::F32Vec::load_array(
        d,
        &data[row_stride * (198 % D::F32Vec::LEN) + (198 / D::F32Vec::LEN)],
    );
    let mut v199 = D::F32Vec::load_array(
        d,
        &data[row_stride * (199 % D::F32Vec::LEN) + (199 / D::F32Vec::LEN)],
    );
    let mut v200 = D::F32Vec::load_array(
        d,
        &data[row_stride * (200 % D::F32Vec::LEN) + (200 / D::F32Vec::LEN)],
    );
    let mut v201 = D::F32Vec::load_array(
        d,
        &data[row_stride * (201 % D::F32Vec::LEN) + (201 / D::F32Vec::LEN)],
    );
    let mut v202 = D::F32Vec::load_array(
        d,
        &data[row_stride * (202 % D::F32Vec::LEN) + (202 / D::F32Vec::LEN)],
    );
    let mut v203 = D::F32Vec::load_array(
        d,
        &data[row_stride * (203 % D::F32Vec::LEN) + (203 / D::F32Vec::LEN)],
    );
    let mut v204 = D::F32Vec::load_array(
        d,
        &data[row_stride * (204 % D::F32Vec::LEN) + (204 / D::F32Vec::LEN)],
    );
    let mut v205 = D::F32Vec::load_array(
        d,
        &data[row_stride * (205 % D::F32Vec::LEN) + (205 / D::F32Vec::LEN)],
    );
    let mut v206 = D::F32Vec::load_array(
        d,
        &data[row_stride * (206 % D::F32Vec::LEN) + (206 / D::F32Vec::LEN)],
    );
    let mut v207 = D::F32Vec::load_array(
        d,
        &data[row_stride * (207 % D::F32Vec::LEN) + (207 / D::F32Vec::LEN)],
    );
    let mut v208 = D::F32Vec::load_array(
        d,
        &data[row_stride * (208 % D::F32Vec::LEN) + (208 / D::F32Vec::LEN)],
    );
    let mut v209 = D::F32Vec::load_array(
        d,
        &data[row_stride * (209 % D::F32Vec::LEN) + (209 / D::F32Vec::LEN)],
    );
    let mut v210 = D::F32Vec::load_array(
        d,
        &data[row_stride * (210 % D::F32Vec::LEN) + (210 / D::F32Vec::LEN)],
    );
    let mut v211 = D::F32Vec::load_array(
        d,
        &data[row_stride * (211 % D::F32Vec::LEN) + (211 / D::F32Vec::LEN)],
    );
    let mut v212 = D::F32Vec::load_array(
        d,
        &data[row_stride * (212 % D::F32Vec::LEN) + (212 / D::F32Vec::LEN)],
    );
    let mut v213 = D::F32Vec::load_array(
        d,
        &data[row_stride * (213 % D::F32Vec::LEN) + (213 / D::F32Vec::LEN)],
    );
    let mut v214 = D::F32Vec::load_array(
        d,
        &data[row_stride * (214 % D::F32Vec::LEN) + (214 / D::F32Vec::LEN)],
    );
    let mut v215 = D::F32Vec::load_array(
        d,
        &data[row_stride * (215 % D::F32Vec::LEN) + (215 / D::F32Vec::LEN)],
    );
    let mut v216 = D::F32Vec::load_array(
        d,
        &data[row_stride * (216 % D::F32Vec::LEN) + (216 / D::F32Vec::LEN)],
    );
    let mut v217 = D::F32Vec::load_array(
        d,
        &data[row_stride * (217 % D::F32Vec::LEN) + (217 / D::F32Vec::LEN)],
    );
    let mut v218 = D::F32Vec::load_array(
        d,
        &data[row_stride * (218 % D::F32Vec::LEN) + (218 / D::F32Vec::LEN)],
    );
    let mut v219 = D::F32Vec::load_array(
        d,
        &data[row_stride * (219 % D::F32Vec::LEN) + (219 / D::F32Vec::LEN)],
    );
    let mut v220 = D::F32Vec::load_array(
        d,
        &data[row_stride * (220 % D::F32Vec::LEN) + (220 / D::F32Vec::LEN)],
    );
    let mut v221 = D::F32Vec::load_array(
        d,
        &data[row_stride * (221 % D::F32Vec::LEN) + (221 / D::F32Vec::LEN)],
    );
    let mut v222 = D::F32Vec::load_array(
        d,
        &data[row_stride * (222 % D::F32Vec::LEN) + (222 / D::F32Vec::LEN)],
    );
    let mut v223 = D::F32Vec::load_array(
        d,
        &data[row_stride * (223 % D::F32Vec::LEN) + (223 / D::F32Vec::LEN)],
    );
    let mut v224 = D::F32Vec::load_array(
        d,
        &data[row_stride * (224 % D::F32Vec::LEN) + (224 / D::F32Vec::LEN)],
    );
    let mut v225 = D::F32Vec::load_array(
        d,
        &data[row_stride * (225 % D::F32Vec::LEN) + (225 / D::F32Vec::LEN)],
    );
    let mut v226 = D::F32Vec::load_array(
        d,
        &data[row_stride * (226 % D::F32Vec::LEN) + (226 / D::F32Vec::LEN)],
    );
    let mut v227 = D::F32Vec::load_array(
        d,
        &data[row_stride * (227 % D::F32Vec::LEN) + (227 / D::F32Vec::LEN)],
    );
    let mut v228 = D::F32Vec::load_array(
        d,
        &data[row_stride * (228 % D::F32Vec::LEN) + (228 / D::F32Vec::LEN)],
    );
    let mut v229 = D::F32Vec::load_array(
        d,
        &data[row_stride * (229 % D::F32Vec::LEN) + (229 / D::F32Vec::LEN)],
    );
    let mut v230 = D::F32Vec::load_array(
        d,
        &data[row_stride * (230 % D::F32Vec::LEN) + (230 / D::F32Vec::LEN)],
    );
    let mut v231 = D::F32Vec::load_array(
        d,
        &data[row_stride * (231 % D::F32Vec::LEN) + (231 / D::F32Vec::LEN)],
    );
    let mut v232 = D::F32Vec::load_array(
        d,
        &data[row_stride * (232 % D::F32Vec::LEN) + (232 / D::F32Vec::LEN)],
    );
    let mut v233 = D::F32Vec::load_array(
        d,
        &data[row_stride * (233 % D::F32Vec::LEN) + (233 / D::F32Vec::LEN)],
    );
    let mut v234 = D::F32Vec::load_array(
        d,
        &data[row_stride * (234 % D::F32Vec::LEN) + (234 / D::F32Vec::LEN)],
    );
    let mut v235 = D::F32Vec::load_array(
        d,
        &data[row_stride * (235 % D::F32Vec::LEN) + (235 / D::F32Vec::LEN)],
    );
    let mut v236 = D::F32Vec::load_array(
        d,
        &data[row_stride * (236 % D::F32Vec::LEN) + (236 / D::F32Vec::LEN)],
    );
    let mut v237 = D::F32Vec::load_array(
        d,
        &data[row_stride * (237 % D::F32Vec::LEN) + (237 / D::F32Vec::LEN)],
    );
    let mut v238 = D::F32Vec::load_array(
        d,
        &data[row_stride * (238 % D::F32Vec::LEN) + (238 / D::F32Vec::LEN)],
    );
    let mut v239 = D::F32Vec::load_array(
        d,
        &data[row_stride * (239 % D::F32Vec::LEN) + (239 / D::F32Vec::LEN)],
    );
    let mut v240 = D::F32Vec::load_array(
        d,
        &data[row_stride * (240 % D::F32Vec::LEN) + (240 / D::F32Vec::LEN)],
    );
    let mut v241 = D::F32Vec::load_array(
        d,
        &data[row_stride * (241 % D::F32Vec::LEN) + (241 / D::F32Vec::LEN)],
    );
    let mut v242 = D::F32Vec::load_array(
        d,
        &data[row_stride * (242 % D::F32Vec::LEN) + (242 / D::F32Vec::LEN)],
    );
    let mut v243 = D::F32Vec::load_array(
        d,
        &data[row_stride * (243 % D::F32Vec::LEN) + (243 / D::F32Vec::LEN)],
    );
    let mut v244 = D::F32Vec::load_array(
        d,
        &data[row_stride * (244 % D::F32Vec::LEN) + (244 / D::F32Vec::LEN)],
    );
    let mut v245 = D::F32Vec::load_array(
        d,
        &data[row_stride * (245 % D::F32Vec::LEN) + (245 / D::F32Vec::LEN)],
    );
    let mut v246 = D::F32Vec::load_array(
        d,
        &data[row_stride * (246 % D::F32Vec::LEN) + (246 / D::F32Vec::LEN)],
    );
    let mut v247 = D::F32Vec::load_array(
        d,
        &data[row_stride * (247 % D::F32Vec::LEN) + (247 / D::F32Vec::LEN)],
    );
    let mut v248 = D::F32Vec::load_array(
        d,
        &data[row_stride * (248 % D::F32Vec::LEN) + (248 / D::F32Vec::LEN)],
    );
    let mut v249 = D::F32Vec::load_array(
        d,
        &data[row_stride * (249 % D::F32Vec::LEN) + (249 / D::F32Vec::LEN)],
    );
    let mut v250 = D::F32Vec::load_array(
        d,
        &data[row_stride * (250 % D::F32Vec::LEN) + (250 / D::F32Vec::LEN)],
    );
    let mut v251 = D::F32Vec::load_array(
        d,
        &data[row_stride * (251 % D::F32Vec::LEN) + (251 / D::F32Vec::LEN)],
    );
    let mut v252 = D::F32Vec::load_array(
        d,
        &data[row_stride * (252 % D::F32Vec::LEN) + (252 / D::F32Vec::LEN)],
    );
    let mut v253 = D::F32Vec::load_array(
        d,
        &data[row_stride * (253 % D::F32Vec::LEN) + (253 / D::F32Vec::LEN)],
    );
    let mut v254 = D::F32Vec::load_array(
        d,
        &data[row_stride * (254 % D::F32Vec::LEN) + (254 / D::F32Vec::LEN)],
    );
    let mut v255 = D::F32Vec::load_array(
        d,
        &data[row_stride * (255 % D::F32Vec::LEN) + (255 / D::F32Vec::LEN)],
    );
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73,
        v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91,
        v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107,
        v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122,
        v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137,
        v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152,
        v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166, v167,
        v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181, v182,
        v183, v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194, v195, v196, v197,
        v198, v199, v200, v201, v202, v203, v204, v205, v206, v207, v208, v209, v210, v211, v212,
        v213, v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227,
        v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241, v242,
        v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255,
    ) = idct_256(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72,
        v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90,
        v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106,
        v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121,
        v122, v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136,
        v137, v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151,
        v152, v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166,
        v167, v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181,
        v182, v183, v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194, v195, v196,
        v197, v198, v199, v200, v201, v202, v203, v204, v205, v206, v207, v208, v209, v210, v211,
        v212, v213, v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226,
        v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241,
        v242, v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255,
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
    v128.store_array(&mut data[row_stride * (128 % D::F32Vec::LEN) + (128 / D::F32Vec::LEN)]);
    v129.store_array(&mut data[row_stride * (129 % D::F32Vec::LEN) + (129 / D::F32Vec::LEN)]);
    v130.store_array(&mut data[row_stride * (130 % D::F32Vec::LEN) + (130 / D::F32Vec::LEN)]);
    v131.store_array(&mut data[row_stride * (131 % D::F32Vec::LEN) + (131 / D::F32Vec::LEN)]);
    v132.store_array(&mut data[row_stride * (132 % D::F32Vec::LEN) + (132 / D::F32Vec::LEN)]);
    v133.store_array(&mut data[row_stride * (133 % D::F32Vec::LEN) + (133 / D::F32Vec::LEN)]);
    v134.store_array(&mut data[row_stride * (134 % D::F32Vec::LEN) + (134 / D::F32Vec::LEN)]);
    v135.store_array(&mut data[row_stride * (135 % D::F32Vec::LEN) + (135 / D::F32Vec::LEN)]);
    v136.store_array(&mut data[row_stride * (136 % D::F32Vec::LEN) + (136 / D::F32Vec::LEN)]);
    v137.store_array(&mut data[row_stride * (137 % D::F32Vec::LEN) + (137 / D::F32Vec::LEN)]);
    v138.store_array(&mut data[row_stride * (138 % D::F32Vec::LEN) + (138 / D::F32Vec::LEN)]);
    v139.store_array(&mut data[row_stride * (139 % D::F32Vec::LEN) + (139 / D::F32Vec::LEN)]);
    v140.store_array(&mut data[row_stride * (140 % D::F32Vec::LEN) + (140 / D::F32Vec::LEN)]);
    v141.store_array(&mut data[row_stride * (141 % D::F32Vec::LEN) + (141 / D::F32Vec::LEN)]);
    v142.store_array(&mut data[row_stride * (142 % D::F32Vec::LEN) + (142 / D::F32Vec::LEN)]);
    v143.store_array(&mut data[row_stride * (143 % D::F32Vec::LEN) + (143 / D::F32Vec::LEN)]);
    v144.store_array(&mut data[row_stride * (144 % D::F32Vec::LEN) + (144 / D::F32Vec::LEN)]);
    v145.store_array(&mut data[row_stride * (145 % D::F32Vec::LEN) + (145 / D::F32Vec::LEN)]);
    v146.store_array(&mut data[row_stride * (146 % D::F32Vec::LEN) + (146 / D::F32Vec::LEN)]);
    v147.store_array(&mut data[row_stride * (147 % D::F32Vec::LEN) + (147 / D::F32Vec::LEN)]);
    v148.store_array(&mut data[row_stride * (148 % D::F32Vec::LEN) + (148 / D::F32Vec::LEN)]);
    v149.store_array(&mut data[row_stride * (149 % D::F32Vec::LEN) + (149 / D::F32Vec::LEN)]);
    v150.store_array(&mut data[row_stride * (150 % D::F32Vec::LEN) + (150 / D::F32Vec::LEN)]);
    v151.store_array(&mut data[row_stride * (151 % D::F32Vec::LEN) + (151 / D::F32Vec::LEN)]);
    v152.store_array(&mut data[row_stride * (152 % D::F32Vec::LEN) + (152 / D::F32Vec::LEN)]);
    v153.store_array(&mut data[row_stride * (153 % D::F32Vec::LEN) + (153 / D::F32Vec::LEN)]);
    v154.store_array(&mut data[row_stride * (154 % D::F32Vec::LEN) + (154 / D::F32Vec::LEN)]);
    v155.store_array(&mut data[row_stride * (155 % D::F32Vec::LEN) + (155 / D::F32Vec::LEN)]);
    v156.store_array(&mut data[row_stride * (156 % D::F32Vec::LEN) + (156 / D::F32Vec::LEN)]);
    v157.store_array(&mut data[row_stride * (157 % D::F32Vec::LEN) + (157 / D::F32Vec::LEN)]);
    v158.store_array(&mut data[row_stride * (158 % D::F32Vec::LEN) + (158 / D::F32Vec::LEN)]);
    v159.store_array(&mut data[row_stride * (159 % D::F32Vec::LEN) + (159 / D::F32Vec::LEN)]);
    v160.store_array(&mut data[row_stride * (160 % D::F32Vec::LEN) + (160 / D::F32Vec::LEN)]);
    v161.store_array(&mut data[row_stride * (161 % D::F32Vec::LEN) + (161 / D::F32Vec::LEN)]);
    v162.store_array(&mut data[row_stride * (162 % D::F32Vec::LEN) + (162 / D::F32Vec::LEN)]);
    v163.store_array(&mut data[row_stride * (163 % D::F32Vec::LEN) + (163 / D::F32Vec::LEN)]);
    v164.store_array(&mut data[row_stride * (164 % D::F32Vec::LEN) + (164 / D::F32Vec::LEN)]);
    v165.store_array(&mut data[row_stride * (165 % D::F32Vec::LEN) + (165 / D::F32Vec::LEN)]);
    v166.store_array(&mut data[row_stride * (166 % D::F32Vec::LEN) + (166 / D::F32Vec::LEN)]);
    v167.store_array(&mut data[row_stride * (167 % D::F32Vec::LEN) + (167 / D::F32Vec::LEN)]);
    v168.store_array(&mut data[row_stride * (168 % D::F32Vec::LEN) + (168 / D::F32Vec::LEN)]);
    v169.store_array(&mut data[row_stride * (169 % D::F32Vec::LEN) + (169 / D::F32Vec::LEN)]);
    v170.store_array(&mut data[row_stride * (170 % D::F32Vec::LEN) + (170 / D::F32Vec::LEN)]);
    v171.store_array(&mut data[row_stride * (171 % D::F32Vec::LEN) + (171 / D::F32Vec::LEN)]);
    v172.store_array(&mut data[row_stride * (172 % D::F32Vec::LEN) + (172 / D::F32Vec::LEN)]);
    v173.store_array(&mut data[row_stride * (173 % D::F32Vec::LEN) + (173 / D::F32Vec::LEN)]);
    v174.store_array(&mut data[row_stride * (174 % D::F32Vec::LEN) + (174 / D::F32Vec::LEN)]);
    v175.store_array(&mut data[row_stride * (175 % D::F32Vec::LEN) + (175 / D::F32Vec::LEN)]);
    v176.store_array(&mut data[row_stride * (176 % D::F32Vec::LEN) + (176 / D::F32Vec::LEN)]);
    v177.store_array(&mut data[row_stride * (177 % D::F32Vec::LEN) + (177 / D::F32Vec::LEN)]);
    v178.store_array(&mut data[row_stride * (178 % D::F32Vec::LEN) + (178 / D::F32Vec::LEN)]);
    v179.store_array(&mut data[row_stride * (179 % D::F32Vec::LEN) + (179 / D::F32Vec::LEN)]);
    v180.store_array(&mut data[row_stride * (180 % D::F32Vec::LEN) + (180 / D::F32Vec::LEN)]);
    v181.store_array(&mut data[row_stride * (181 % D::F32Vec::LEN) + (181 / D::F32Vec::LEN)]);
    v182.store_array(&mut data[row_stride * (182 % D::F32Vec::LEN) + (182 / D::F32Vec::LEN)]);
    v183.store_array(&mut data[row_stride * (183 % D::F32Vec::LEN) + (183 / D::F32Vec::LEN)]);
    v184.store_array(&mut data[row_stride * (184 % D::F32Vec::LEN) + (184 / D::F32Vec::LEN)]);
    v185.store_array(&mut data[row_stride * (185 % D::F32Vec::LEN) + (185 / D::F32Vec::LEN)]);
    v186.store_array(&mut data[row_stride * (186 % D::F32Vec::LEN) + (186 / D::F32Vec::LEN)]);
    v187.store_array(&mut data[row_stride * (187 % D::F32Vec::LEN) + (187 / D::F32Vec::LEN)]);
    v188.store_array(&mut data[row_stride * (188 % D::F32Vec::LEN) + (188 / D::F32Vec::LEN)]);
    v189.store_array(&mut data[row_stride * (189 % D::F32Vec::LEN) + (189 / D::F32Vec::LEN)]);
    v190.store_array(&mut data[row_stride * (190 % D::F32Vec::LEN) + (190 / D::F32Vec::LEN)]);
    v191.store_array(&mut data[row_stride * (191 % D::F32Vec::LEN) + (191 / D::F32Vec::LEN)]);
    v192.store_array(&mut data[row_stride * (192 % D::F32Vec::LEN) + (192 / D::F32Vec::LEN)]);
    v193.store_array(&mut data[row_stride * (193 % D::F32Vec::LEN) + (193 / D::F32Vec::LEN)]);
    v194.store_array(&mut data[row_stride * (194 % D::F32Vec::LEN) + (194 / D::F32Vec::LEN)]);
    v195.store_array(&mut data[row_stride * (195 % D::F32Vec::LEN) + (195 / D::F32Vec::LEN)]);
    v196.store_array(&mut data[row_stride * (196 % D::F32Vec::LEN) + (196 / D::F32Vec::LEN)]);
    v197.store_array(&mut data[row_stride * (197 % D::F32Vec::LEN) + (197 / D::F32Vec::LEN)]);
    v198.store_array(&mut data[row_stride * (198 % D::F32Vec::LEN) + (198 / D::F32Vec::LEN)]);
    v199.store_array(&mut data[row_stride * (199 % D::F32Vec::LEN) + (199 / D::F32Vec::LEN)]);
    v200.store_array(&mut data[row_stride * (200 % D::F32Vec::LEN) + (200 / D::F32Vec::LEN)]);
    v201.store_array(&mut data[row_stride * (201 % D::F32Vec::LEN) + (201 / D::F32Vec::LEN)]);
    v202.store_array(&mut data[row_stride * (202 % D::F32Vec::LEN) + (202 / D::F32Vec::LEN)]);
    v203.store_array(&mut data[row_stride * (203 % D::F32Vec::LEN) + (203 / D::F32Vec::LEN)]);
    v204.store_array(&mut data[row_stride * (204 % D::F32Vec::LEN) + (204 / D::F32Vec::LEN)]);
    v205.store_array(&mut data[row_stride * (205 % D::F32Vec::LEN) + (205 / D::F32Vec::LEN)]);
    v206.store_array(&mut data[row_stride * (206 % D::F32Vec::LEN) + (206 / D::F32Vec::LEN)]);
    v207.store_array(&mut data[row_stride * (207 % D::F32Vec::LEN) + (207 / D::F32Vec::LEN)]);
    v208.store_array(&mut data[row_stride * (208 % D::F32Vec::LEN) + (208 / D::F32Vec::LEN)]);
    v209.store_array(&mut data[row_stride * (209 % D::F32Vec::LEN) + (209 / D::F32Vec::LEN)]);
    v210.store_array(&mut data[row_stride * (210 % D::F32Vec::LEN) + (210 / D::F32Vec::LEN)]);
    v211.store_array(&mut data[row_stride * (211 % D::F32Vec::LEN) + (211 / D::F32Vec::LEN)]);
    v212.store_array(&mut data[row_stride * (212 % D::F32Vec::LEN) + (212 / D::F32Vec::LEN)]);
    v213.store_array(&mut data[row_stride * (213 % D::F32Vec::LEN) + (213 / D::F32Vec::LEN)]);
    v214.store_array(&mut data[row_stride * (214 % D::F32Vec::LEN) + (214 / D::F32Vec::LEN)]);
    v215.store_array(&mut data[row_stride * (215 % D::F32Vec::LEN) + (215 / D::F32Vec::LEN)]);
    v216.store_array(&mut data[row_stride * (216 % D::F32Vec::LEN) + (216 / D::F32Vec::LEN)]);
    v217.store_array(&mut data[row_stride * (217 % D::F32Vec::LEN) + (217 / D::F32Vec::LEN)]);
    v218.store_array(&mut data[row_stride * (218 % D::F32Vec::LEN) + (218 / D::F32Vec::LEN)]);
    v219.store_array(&mut data[row_stride * (219 % D::F32Vec::LEN) + (219 / D::F32Vec::LEN)]);
    v220.store_array(&mut data[row_stride * (220 % D::F32Vec::LEN) + (220 / D::F32Vec::LEN)]);
    v221.store_array(&mut data[row_stride * (221 % D::F32Vec::LEN) + (221 / D::F32Vec::LEN)]);
    v222.store_array(&mut data[row_stride * (222 % D::F32Vec::LEN) + (222 / D::F32Vec::LEN)]);
    v223.store_array(&mut data[row_stride * (223 % D::F32Vec::LEN) + (223 / D::F32Vec::LEN)]);
    v224.store_array(&mut data[row_stride * (224 % D::F32Vec::LEN) + (224 / D::F32Vec::LEN)]);
    v225.store_array(&mut data[row_stride * (225 % D::F32Vec::LEN) + (225 / D::F32Vec::LEN)]);
    v226.store_array(&mut data[row_stride * (226 % D::F32Vec::LEN) + (226 / D::F32Vec::LEN)]);
    v227.store_array(&mut data[row_stride * (227 % D::F32Vec::LEN) + (227 / D::F32Vec::LEN)]);
    v228.store_array(&mut data[row_stride * (228 % D::F32Vec::LEN) + (228 / D::F32Vec::LEN)]);
    v229.store_array(&mut data[row_stride * (229 % D::F32Vec::LEN) + (229 / D::F32Vec::LEN)]);
    v230.store_array(&mut data[row_stride * (230 % D::F32Vec::LEN) + (230 / D::F32Vec::LEN)]);
    v231.store_array(&mut data[row_stride * (231 % D::F32Vec::LEN) + (231 / D::F32Vec::LEN)]);
    v232.store_array(&mut data[row_stride * (232 % D::F32Vec::LEN) + (232 / D::F32Vec::LEN)]);
    v233.store_array(&mut data[row_stride * (233 % D::F32Vec::LEN) + (233 / D::F32Vec::LEN)]);
    v234.store_array(&mut data[row_stride * (234 % D::F32Vec::LEN) + (234 / D::F32Vec::LEN)]);
    v235.store_array(&mut data[row_stride * (235 % D::F32Vec::LEN) + (235 / D::F32Vec::LEN)]);
    v236.store_array(&mut data[row_stride * (236 % D::F32Vec::LEN) + (236 / D::F32Vec::LEN)]);
    v237.store_array(&mut data[row_stride * (237 % D::F32Vec::LEN) + (237 / D::F32Vec::LEN)]);
    v238.store_array(&mut data[row_stride * (238 % D::F32Vec::LEN) + (238 / D::F32Vec::LEN)]);
    v239.store_array(&mut data[row_stride * (239 % D::F32Vec::LEN) + (239 / D::F32Vec::LEN)]);
    v240.store_array(&mut data[row_stride * (240 % D::F32Vec::LEN) + (240 / D::F32Vec::LEN)]);
    v241.store_array(&mut data[row_stride * (241 % D::F32Vec::LEN) + (241 / D::F32Vec::LEN)]);
    v242.store_array(&mut data[row_stride * (242 % D::F32Vec::LEN) + (242 / D::F32Vec::LEN)]);
    v243.store_array(&mut data[row_stride * (243 % D::F32Vec::LEN) + (243 / D::F32Vec::LEN)]);
    v244.store_array(&mut data[row_stride * (244 % D::F32Vec::LEN) + (244 / D::F32Vec::LEN)]);
    v245.store_array(&mut data[row_stride * (245 % D::F32Vec::LEN) + (245 / D::F32Vec::LEN)]);
    v246.store_array(&mut data[row_stride * (246 % D::F32Vec::LEN) + (246 / D::F32Vec::LEN)]);
    v247.store_array(&mut data[row_stride * (247 % D::F32Vec::LEN) + (247 / D::F32Vec::LEN)]);
    v248.store_array(&mut data[row_stride * (248 % D::F32Vec::LEN) + (248 / D::F32Vec::LEN)]);
    v249.store_array(&mut data[row_stride * (249 % D::F32Vec::LEN) + (249 / D::F32Vec::LEN)]);
    v250.store_array(&mut data[row_stride * (250 % D::F32Vec::LEN) + (250 / D::F32Vec::LEN)]);
    v251.store_array(&mut data[row_stride * (251 % D::F32Vec::LEN) + (251 / D::F32Vec::LEN)]);
    v252.store_array(&mut data[row_stride * (252 % D::F32Vec::LEN) + (252 / D::F32Vec::LEN)]);
    v253.store_array(&mut data[row_stride * (253 % D::F32Vec::LEN) + (253 / D::F32Vec::LEN)]);
    v254.store_array(&mut data[row_stride * (254 % D::F32Vec::LEN) + (254 / D::F32Vec::LEN)]);
    v255.store_array(&mut data[row_stride * (255 % D::F32Vec::LEN) + (255 / D::F32Vec::LEN)]);
}

#[inline(always)]
pub(super) fn do_idct_256_trh<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
) {
    let row_stride = 128 / D::F32Vec::LEN;
    assert!(data.len() > 255 * row_stride);
    const { assert!(128usize.is_multiple_of(D::F32Vec::LEN)) };
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
    let mut v64 = D::F32Vec::load_array(d, &data[row_stride * 128]);
    let mut v65 = D::F32Vec::load_array(d, &data[row_stride * 130]);
    let mut v66 = D::F32Vec::load_array(d, &data[row_stride * 132]);
    let mut v67 = D::F32Vec::load_array(d, &data[row_stride * 134]);
    let mut v68 = D::F32Vec::load_array(d, &data[row_stride * 136]);
    let mut v69 = D::F32Vec::load_array(d, &data[row_stride * 138]);
    let mut v70 = D::F32Vec::load_array(d, &data[row_stride * 140]);
    let mut v71 = D::F32Vec::load_array(d, &data[row_stride * 142]);
    let mut v72 = D::F32Vec::load_array(d, &data[row_stride * 144]);
    let mut v73 = D::F32Vec::load_array(d, &data[row_stride * 146]);
    let mut v74 = D::F32Vec::load_array(d, &data[row_stride * 148]);
    let mut v75 = D::F32Vec::load_array(d, &data[row_stride * 150]);
    let mut v76 = D::F32Vec::load_array(d, &data[row_stride * 152]);
    let mut v77 = D::F32Vec::load_array(d, &data[row_stride * 154]);
    let mut v78 = D::F32Vec::load_array(d, &data[row_stride * 156]);
    let mut v79 = D::F32Vec::load_array(d, &data[row_stride * 158]);
    let mut v80 = D::F32Vec::load_array(d, &data[row_stride * 160]);
    let mut v81 = D::F32Vec::load_array(d, &data[row_stride * 162]);
    let mut v82 = D::F32Vec::load_array(d, &data[row_stride * 164]);
    let mut v83 = D::F32Vec::load_array(d, &data[row_stride * 166]);
    let mut v84 = D::F32Vec::load_array(d, &data[row_stride * 168]);
    let mut v85 = D::F32Vec::load_array(d, &data[row_stride * 170]);
    let mut v86 = D::F32Vec::load_array(d, &data[row_stride * 172]);
    let mut v87 = D::F32Vec::load_array(d, &data[row_stride * 174]);
    let mut v88 = D::F32Vec::load_array(d, &data[row_stride * 176]);
    let mut v89 = D::F32Vec::load_array(d, &data[row_stride * 178]);
    let mut v90 = D::F32Vec::load_array(d, &data[row_stride * 180]);
    let mut v91 = D::F32Vec::load_array(d, &data[row_stride * 182]);
    let mut v92 = D::F32Vec::load_array(d, &data[row_stride * 184]);
    let mut v93 = D::F32Vec::load_array(d, &data[row_stride * 186]);
    let mut v94 = D::F32Vec::load_array(d, &data[row_stride * 188]);
    let mut v95 = D::F32Vec::load_array(d, &data[row_stride * 190]);
    let mut v96 = D::F32Vec::load_array(d, &data[row_stride * 192]);
    let mut v97 = D::F32Vec::load_array(d, &data[row_stride * 194]);
    let mut v98 = D::F32Vec::load_array(d, &data[row_stride * 196]);
    let mut v99 = D::F32Vec::load_array(d, &data[row_stride * 198]);
    let mut v100 = D::F32Vec::load_array(d, &data[row_stride * 200]);
    let mut v101 = D::F32Vec::load_array(d, &data[row_stride * 202]);
    let mut v102 = D::F32Vec::load_array(d, &data[row_stride * 204]);
    let mut v103 = D::F32Vec::load_array(d, &data[row_stride * 206]);
    let mut v104 = D::F32Vec::load_array(d, &data[row_stride * 208]);
    let mut v105 = D::F32Vec::load_array(d, &data[row_stride * 210]);
    let mut v106 = D::F32Vec::load_array(d, &data[row_stride * 212]);
    let mut v107 = D::F32Vec::load_array(d, &data[row_stride * 214]);
    let mut v108 = D::F32Vec::load_array(d, &data[row_stride * 216]);
    let mut v109 = D::F32Vec::load_array(d, &data[row_stride * 218]);
    let mut v110 = D::F32Vec::load_array(d, &data[row_stride * 220]);
    let mut v111 = D::F32Vec::load_array(d, &data[row_stride * 222]);
    let mut v112 = D::F32Vec::load_array(d, &data[row_stride * 224]);
    let mut v113 = D::F32Vec::load_array(d, &data[row_stride * 226]);
    let mut v114 = D::F32Vec::load_array(d, &data[row_stride * 228]);
    let mut v115 = D::F32Vec::load_array(d, &data[row_stride * 230]);
    let mut v116 = D::F32Vec::load_array(d, &data[row_stride * 232]);
    let mut v117 = D::F32Vec::load_array(d, &data[row_stride * 234]);
    let mut v118 = D::F32Vec::load_array(d, &data[row_stride * 236]);
    let mut v119 = D::F32Vec::load_array(d, &data[row_stride * 238]);
    let mut v120 = D::F32Vec::load_array(d, &data[row_stride * 240]);
    let mut v121 = D::F32Vec::load_array(d, &data[row_stride * 242]);
    let mut v122 = D::F32Vec::load_array(d, &data[row_stride * 244]);
    let mut v123 = D::F32Vec::load_array(d, &data[row_stride * 246]);
    let mut v124 = D::F32Vec::load_array(d, &data[row_stride * 248]);
    let mut v125 = D::F32Vec::load_array(d, &data[row_stride * 250]);
    let mut v126 = D::F32Vec::load_array(d, &data[row_stride * 252]);
    let mut v127 = D::F32Vec::load_array(d, &data[row_stride * 254]);
    let mut v128 = D::F32Vec::load_array(d, &data[row_stride * 1]);
    let mut v129 = D::F32Vec::load_array(d, &data[row_stride * 3]);
    let mut v130 = D::F32Vec::load_array(d, &data[row_stride * 5]);
    let mut v131 = D::F32Vec::load_array(d, &data[row_stride * 7]);
    let mut v132 = D::F32Vec::load_array(d, &data[row_stride * 9]);
    let mut v133 = D::F32Vec::load_array(d, &data[row_stride * 11]);
    let mut v134 = D::F32Vec::load_array(d, &data[row_stride * 13]);
    let mut v135 = D::F32Vec::load_array(d, &data[row_stride * 15]);
    let mut v136 = D::F32Vec::load_array(d, &data[row_stride * 17]);
    let mut v137 = D::F32Vec::load_array(d, &data[row_stride * 19]);
    let mut v138 = D::F32Vec::load_array(d, &data[row_stride * 21]);
    let mut v139 = D::F32Vec::load_array(d, &data[row_stride * 23]);
    let mut v140 = D::F32Vec::load_array(d, &data[row_stride * 25]);
    let mut v141 = D::F32Vec::load_array(d, &data[row_stride * 27]);
    let mut v142 = D::F32Vec::load_array(d, &data[row_stride * 29]);
    let mut v143 = D::F32Vec::load_array(d, &data[row_stride * 31]);
    let mut v144 = D::F32Vec::load_array(d, &data[row_stride * 33]);
    let mut v145 = D::F32Vec::load_array(d, &data[row_stride * 35]);
    let mut v146 = D::F32Vec::load_array(d, &data[row_stride * 37]);
    let mut v147 = D::F32Vec::load_array(d, &data[row_stride * 39]);
    let mut v148 = D::F32Vec::load_array(d, &data[row_stride * 41]);
    let mut v149 = D::F32Vec::load_array(d, &data[row_stride * 43]);
    let mut v150 = D::F32Vec::load_array(d, &data[row_stride * 45]);
    let mut v151 = D::F32Vec::load_array(d, &data[row_stride * 47]);
    let mut v152 = D::F32Vec::load_array(d, &data[row_stride * 49]);
    let mut v153 = D::F32Vec::load_array(d, &data[row_stride * 51]);
    let mut v154 = D::F32Vec::load_array(d, &data[row_stride * 53]);
    let mut v155 = D::F32Vec::load_array(d, &data[row_stride * 55]);
    let mut v156 = D::F32Vec::load_array(d, &data[row_stride * 57]);
    let mut v157 = D::F32Vec::load_array(d, &data[row_stride * 59]);
    let mut v158 = D::F32Vec::load_array(d, &data[row_stride * 61]);
    let mut v159 = D::F32Vec::load_array(d, &data[row_stride * 63]);
    let mut v160 = D::F32Vec::load_array(d, &data[row_stride * 65]);
    let mut v161 = D::F32Vec::load_array(d, &data[row_stride * 67]);
    let mut v162 = D::F32Vec::load_array(d, &data[row_stride * 69]);
    let mut v163 = D::F32Vec::load_array(d, &data[row_stride * 71]);
    let mut v164 = D::F32Vec::load_array(d, &data[row_stride * 73]);
    let mut v165 = D::F32Vec::load_array(d, &data[row_stride * 75]);
    let mut v166 = D::F32Vec::load_array(d, &data[row_stride * 77]);
    let mut v167 = D::F32Vec::load_array(d, &data[row_stride * 79]);
    let mut v168 = D::F32Vec::load_array(d, &data[row_stride * 81]);
    let mut v169 = D::F32Vec::load_array(d, &data[row_stride * 83]);
    let mut v170 = D::F32Vec::load_array(d, &data[row_stride * 85]);
    let mut v171 = D::F32Vec::load_array(d, &data[row_stride * 87]);
    let mut v172 = D::F32Vec::load_array(d, &data[row_stride * 89]);
    let mut v173 = D::F32Vec::load_array(d, &data[row_stride * 91]);
    let mut v174 = D::F32Vec::load_array(d, &data[row_stride * 93]);
    let mut v175 = D::F32Vec::load_array(d, &data[row_stride * 95]);
    let mut v176 = D::F32Vec::load_array(d, &data[row_stride * 97]);
    let mut v177 = D::F32Vec::load_array(d, &data[row_stride * 99]);
    let mut v178 = D::F32Vec::load_array(d, &data[row_stride * 101]);
    let mut v179 = D::F32Vec::load_array(d, &data[row_stride * 103]);
    let mut v180 = D::F32Vec::load_array(d, &data[row_stride * 105]);
    let mut v181 = D::F32Vec::load_array(d, &data[row_stride * 107]);
    let mut v182 = D::F32Vec::load_array(d, &data[row_stride * 109]);
    let mut v183 = D::F32Vec::load_array(d, &data[row_stride * 111]);
    let mut v184 = D::F32Vec::load_array(d, &data[row_stride * 113]);
    let mut v185 = D::F32Vec::load_array(d, &data[row_stride * 115]);
    let mut v186 = D::F32Vec::load_array(d, &data[row_stride * 117]);
    let mut v187 = D::F32Vec::load_array(d, &data[row_stride * 119]);
    let mut v188 = D::F32Vec::load_array(d, &data[row_stride * 121]);
    let mut v189 = D::F32Vec::load_array(d, &data[row_stride * 123]);
    let mut v190 = D::F32Vec::load_array(d, &data[row_stride * 125]);
    let mut v191 = D::F32Vec::load_array(d, &data[row_stride * 127]);
    let mut v192 = D::F32Vec::load_array(d, &data[row_stride * 129]);
    let mut v193 = D::F32Vec::load_array(d, &data[row_stride * 131]);
    let mut v194 = D::F32Vec::load_array(d, &data[row_stride * 133]);
    let mut v195 = D::F32Vec::load_array(d, &data[row_stride * 135]);
    let mut v196 = D::F32Vec::load_array(d, &data[row_stride * 137]);
    let mut v197 = D::F32Vec::load_array(d, &data[row_stride * 139]);
    let mut v198 = D::F32Vec::load_array(d, &data[row_stride * 141]);
    let mut v199 = D::F32Vec::load_array(d, &data[row_stride * 143]);
    let mut v200 = D::F32Vec::load_array(d, &data[row_stride * 145]);
    let mut v201 = D::F32Vec::load_array(d, &data[row_stride * 147]);
    let mut v202 = D::F32Vec::load_array(d, &data[row_stride * 149]);
    let mut v203 = D::F32Vec::load_array(d, &data[row_stride * 151]);
    let mut v204 = D::F32Vec::load_array(d, &data[row_stride * 153]);
    let mut v205 = D::F32Vec::load_array(d, &data[row_stride * 155]);
    let mut v206 = D::F32Vec::load_array(d, &data[row_stride * 157]);
    let mut v207 = D::F32Vec::load_array(d, &data[row_stride * 159]);
    let mut v208 = D::F32Vec::load_array(d, &data[row_stride * 161]);
    let mut v209 = D::F32Vec::load_array(d, &data[row_stride * 163]);
    let mut v210 = D::F32Vec::load_array(d, &data[row_stride * 165]);
    let mut v211 = D::F32Vec::load_array(d, &data[row_stride * 167]);
    let mut v212 = D::F32Vec::load_array(d, &data[row_stride * 169]);
    let mut v213 = D::F32Vec::load_array(d, &data[row_stride * 171]);
    let mut v214 = D::F32Vec::load_array(d, &data[row_stride * 173]);
    let mut v215 = D::F32Vec::load_array(d, &data[row_stride * 175]);
    let mut v216 = D::F32Vec::load_array(d, &data[row_stride * 177]);
    let mut v217 = D::F32Vec::load_array(d, &data[row_stride * 179]);
    let mut v218 = D::F32Vec::load_array(d, &data[row_stride * 181]);
    let mut v219 = D::F32Vec::load_array(d, &data[row_stride * 183]);
    let mut v220 = D::F32Vec::load_array(d, &data[row_stride * 185]);
    let mut v221 = D::F32Vec::load_array(d, &data[row_stride * 187]);
    let mut v222 = D::F32Vec::load_array(d, &data[row_stride * 189]);
    let mut v223 = D::F32Vec::load_array(d, &data[row_stride * 191]);
    let mut v224 = D::F32Vec::load_array(d, &data[row_stride * 193]);
    let mut v225 = D::F32Vec::load_array(d, &data[row_stride * 195]);
    let mut v226 = D::F32Vec::load_array(d, &data[row_stride * 197]);
    let mut v227 = D::F32Vec::load_array(d, &data[row_stride * 199]);
    let mut v228 = D::F32Vec::load_array(d, &data[row_stride * 201]);
    let mut v229 = D::F32Vec::load_array(d, &data[row_stride * 203]);
    let mut v230 = D::F32Vec::load_array(d, &data[row_stride * 205]);
    let mut v231 = D::F32Vec::load_array(d, &data[row_stride * 207]);
    let mut v232 = D::F32Vec::load_array(d, &data[row_stride * 209]);
    let mut v233 = D::F32Vec::load_array(d, &data[row_stride * 211]);
    let mut v234 = D::F32Vec::load_array(d, &data[row_stride * 213]);
    let mut v235 = D::F32Vec::load_array(d, &data[row_stride * 215]);
    let mut v236 = D::F32Vec::load_array(d, &data[row_stride * 217]);
    let mut v237 = D::F32Vec::load_array(d, &data[row_stride * 219]);
    let mut v238 = D::F32Vec::load_array(d, &data[row_stride * 221]);
    let mut v239 = D::F32Vec::load_array(d, &data[row_stride * 223]);
    let mut v240 = D::F32Vec::load_array(d, &data[row_stride * 225]);
    let mut v241 = D::F32Vec::load_array(d, &data[row_stride * 227]);
    let mut v242 = D::F32Vec::load_array(d, &data[row_stride * 229]);
    let mut v243 = D::F32Vec::load_array(d, &data[row_stride * 231]);
    let mut v244 = D::F32Vec::load_array(d, &data[row_stride * 233]);
    let mut v245 = D::F32Vec::load_array(d, &data[row_stride * 235]);
    let mut v246 = D::F32Vec::load_array(d, &data[row_stride * 237]);
    let mut v247 = D::F32Vec::load_array(d, &data[row_stride * 239]);
    let mut v248 = D::F32Vec::load_array(d, &data[row_stride * 241]);
    let mut v249 = D::F32Vec::load_array(d, &data[row_stride * 243]);
    let mut v250 = D::F32Vec::load_array(d, &data[row_stride * 245]);
    let mut v251 = D::F32Vec::load_array(d, &data[row_stride * 247]);
    let mut v252 = D::F32Vec::load_array(d, &data[row_stride * 249]);
    let mut v253 = D::F32Vec::load_array(d, &data[row_stride * 251]);
    let mut v254 = D::F32Vec::load_array(d, &data[row_stride * 253]);
    let mut v255 = D::F32Vec::load_array(d, &data[row_stride * 255]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73,
        v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91,
        v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107,
        v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122,
        v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137,
        v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152,
        v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166, v167,
        v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181, v182,
        v183, v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194, v195, v196, v197,
        v198, v199, v200, v201, v202, v203, v204, v205, v206, v207, v208, v209, v210, v211, v212,
        v213, v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227,
        v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241, v242,
        v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255,
    ) = idct_256(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72,
        v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90,
        v91, v92, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106,
        v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119, v120, v121,
        v122, v123, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136,
        v137, v138, v139, v140, v141, v142, v143, v144, v145, v146, v147, v148, v149, v150, v151,
        v152, v153, v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165, v166,
        v167, v168, v169, v170, v171, v172, v173, v174, v175, v176, v177, v178, v179, v180, v181,
        v182, v183, v184, v185, v186, v187, v188, v189, v190, v191, v192, v193, v194, v195, v196,
        v197, v198, v199, v200, v201, v202, v203, v204, v205, v206, v207, v208, v209, v210, v211,
        v212, v213, v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226,
        v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241,
        v242, v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255,
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
    v128.store_array(&mut data[row_stride * 128]);
    v129.store_array(&mut data[row_stride * 129]);
    v130.store_array(&mut data[row_stride * 130]);
    v131.store_array(&mut data[row_stride * 131]);
    v132.store_array(&mut data[row_stride * 132]);
    v133.store_array(&mut data[row_stride * 133]);
    v134.store_array(&mut data[row_stride * 134]);
    v135.store_array(&mut data[row_stride * 135]);
    v136.store_array(&mut data[row_stride * 136]);
    v137.store_array(&mut data[row_stride * 137]);
    v138.store_array(&mut data[row_stride * 138]);
    v139.store_array(&mut data[row_stride * 139]);
    v140.store_array(&mut data[row_stride * 140]);
    v141.store_array(&mut data[row_stride * 141]);
    v142.store_array(&mut data[row_stride * 142]);
    v143.store_array(&mut data[row_stride * 143]);
    v144.store_array(&mut data[row_stride * 144]);
    v145.store_array(&mut data[row_stride * 145]);
    v146.store_array(&mut data[row_stride * 146]);
    v147.store_array(&mut data[row_stride * 147]);
    v148.store_array(&mut data[row_stride * 148]);
    v149.store_array(&mut data[row_stride * 149]);
    v150.store_array(&mut data[row_stride * 150]);
    v151.store_array(&mut data[row_stride * 151]);
    v152.store_array(&mut data[row_stride * 152]);
    v153.store_array(&mut data[row_stride * 153]);
    v154.store_array(&mut data[row_stride * 154]);
    v155.store_array(&mut data[row_stride * 155]);
    v156.store_array(&mut data[row_stride * 156]);
    v157.store_array(&mut data[row_stride * 157]);
    v158.store_array(&mut data[row_stride * 158]);
    v159.store_array(&mut data[row_stride * 159]);
    v160.store_array(&mut data[row_stride * 160]);
    v161.store_array(&mut data[row_stride * 161]);
    v162.store_array(&mut data[row_stride * 162]);
    v163.store_array(&mut data[row_stride * 163]);
    v164.store_array(&mut data[row_stride * 164]);
    v165.store_array(&mut data[row_stride * 165]);
    v166.store_array(&mut data[row_stride * 166]);
    v167.store_array(&mut data[row_stride * 167]);
    v168.store_array(&mut data[row_stride * 168]);
    v169.store_array(&mut data[row_stride * 169]);
    v170.store_array(&mut data[row_stride * 170]);
    v171.store_array(&mut data[row_stride * 171]);
    v172.store_array(&mut data[row_stride * 172]);
    v173.store_array(&mut data[row_stride * 173]);
    v174.store_array(&mut data[row_stride * 174]);
    v175.store_array(&mut data[row_stride * 175]);
    v176.store_array(&mut data[row_stride * 176]);
    v177.store_array(&mut data[row_stride * 177]);
    v178.store_array(&mut data[row_stride * 178]);
    v179.store_array(&mut data[row_stride * 179]);
    v180.store_array(&mut data[row_stride * 180]);
    v181.store_array(&mut data[row_stride * 181]);
    v182.store_array(&mut data[row_stride * 182]);
    v183.store_array(&mut data[row_stride * 183]);
    v184.store_array(&mut data[row_stride * 184]);
    v185.store_array(&mut data[row_stride * 185]);
    v186.store_array(&mut data[row_stride * 186]);
    v187.store_array(&mut data[row_stride * 187]);
    v188.store_array(&mut data[row_stride * 188]);
    v189.store_array(&mut data[row_stride * 189]);
    v190.store_array(&mut data[row_stride * 190]);
    v191.store_array(&mut data[row_stride * 191]);
    v192.store_array(&mut data[row_stride * 192]);
    v193.store_array(&mut data[row_stride * 193]);
    v194.store_array(&mut data[row_stride * 194]);
    v195.store_array(&mut data[row_stride * 195]);
    v196.store_array(&mut data[row_stride * 196]);
    v197.store_array(&mut data[row_stride * 197]);
    v198.store_array(&mut data[row_stride * 198]);
    v199.store_array(&mut data[row_stride * 199]);
    v200.store_array(&mut data[row_stride * 200]);
    v201.store_array(&mut data[row_stride * 201]);
    v202.store_array(&mut data[row_stride * 202]);
    v203.store_array(&mut data[row_stride * 203]);
    v204.store_array(&mut data[row_stride * 204]);
    v205.store_array(&mut data[row_stride * 205]);
    v206.store_array(&mut data[row_stride * 206]);
    v207.store_array(&mut data[row_stride * 207]);
    v208.store_array(&mut data[row_stride * 208]);
    v209.store_array(&mut data[row_stride * 209]);
    v210.store_array(&mut data[row_stride * 210]);
    v211.store_array(&mut data[row_stride * 211]);
    v212.store_array(&mut data[row_stride * 212]);
    v213.store_array(&mut data[row_stride * 213]);
    v214.store_array(&mut data[row_stride * 214]);
    v215.store_array(&mut data[row_stride * 215]);
    v216.store_array(&mut data[row_stride * 216]);
    v217.store_array(&mut data[row_stride * 217]);
    v218.store_array(&mut data[row_stride * 218]);
    v219.store_array(&mut data[row_stride * 219]);
    v220.store_array(&mut data[row_stride * 220]);
    v221.store_array(&mut data[row_stride * 221]);
    v222.store_array(&mut data[row_stride * 222]);
    v223.store_array(&mut data[row_stride * 223]);
    v224.store_array(&mut data[row_stride * 224]);
    v225.store_array(&mut data[row_stride * 225]);
    v226.store_array(&mut data[row_stride * 226]);
    v227.store_array(&mut data[row_stride * 227]);
    v228.store_array(&mut data[row_stride * 228]);
    v229.store_array(&mut data[row_stride * 229]);
    v230.store_array(&mut data[row_stride * 230]);
    v231.store_array(&mut data[row_stride * 231]);
    v232.store_array(&mut data[row_stride * 232]);
    v233.store_array(&mut data[row_stride * 233]);
    v234.store_array(&mut data[row_stride * 234]);
    v235.store_array(&mut data[row_stride * 235]);
    v236.store_array(&mut data[row_stride * 236]);
    v237.store_array(&mut data[row_stride * 237]);
    v238.store_array(&mut data[row_stride * 238]);
    v239.store_array(&mut data[row_stride * 239]);
    v240.store_array(&mut data[row_stride * 240]);
    v241.store_array(&mut data[row_stride * 241]);
    v242.store_array(&mut data[row_stride * 242]);
    v243.store_array(&mut data[row_stride * 243]);
    v244.store_array(&mut data[row_stride * 244]);
    v245.store_array(&mut data[row_stride * 245]);
    v246.store_array(&mut data[row_stride * 246]);
    v247.store_array(&mut data[row_stride * 247]);
    v248.store_array(&mut data[row_stride * 248]);
    v249.store_array(&mut data[row_stride * 249]);
    v250.store_array(&mut data[row_stride * 250]);
    v251.store_array(&mut data[row_stride * 251]);
    v252.store_array(&mut data[row_stride * 252]);
    v253.store_array(&mut data[row_stride * 253]);
    v254.store_array(&mut data[row_stride * 254]);
    v255.store_array(&mut data[row_stride * 255]);
}
