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
pub(super) fn idct_64<D: SimdDescriptor>(
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
) {
    d.call(
        #[inline(always)]
        |_| {
            (
                v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v32,
                v34, v36, v38, v40, v42, v44, v46, v48, v50, v52, v54, v56, v58, v60, v62,
            ) = idct_32(
                d, v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v32,
                v34, v36, v38, v40, v42, v44, v46, v48, v50, v52, v54, v56, v58, v60, v62,
            );
        },
    );
    let mut v64 = v1 + v3;
    let mut v65 = v3 + v5;
    let mut v66 = v5 + v7;
    let mut v67 = v7 + v9;
    let mut v68 = v9 + v11;
    let mut v69 = v11 + v13;
    let mut v70 = v13 + v15;
    let mut v71 = v15 + v17;
    let mut v72 = v17 + v19;
    let mut v73 = v19 + v21;
    let mut v74 = v21 + v23;
    let mut v75 = v23 + v25;
    let mut v76 = v25 + v27;
    let mut v77 = v27 + v29;
    let mut v78 = v29 + v31;
    let mut v79 = v31 + v33;
    let mut v80 = v33 + v35;
    let mut v81 = v35 + v37;
    let mut v82 = v37 + v39;
    let mut v83 = v39 + v41;
    let mut v84 = v41 + v43;
    let mut v85 = v43 + v45;
    let mut v86 = v45 + v47;
    let mut v87 = v47 + v49;
    let mut v88 = v49 + v51;
    let mut v89 = v51 + v53;
    let mut v90 = v53 + v55;
    let mut v91 = v55 + v57;
    let mut v92 = v57 + v59;
    let mut v93 = v59 + v61;
    let mut v94 = v61 + v63;
    let mut v95 = v1 * D::F32Vec::splat(d, std::f32::consts::SQRT_2);
    d.call(
        #[inline(always)]
        |_| {
            (
                v95, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78,
                v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94,
            ) = idct_32(
                d, v95, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78,
                v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94,
            );
        },
    );
    let mul = D::F32Vec::splat(d, 0.5001506360206510);
    let mut v96 = v95.mul_add(mul, v0);
    let mut v97 = v95.neg_mul_add(mul, v0);
    let mul = D::F32Vec::splat(d, 0.5013584524464084);
    let mut v98 = v64.mul_add(mul, v2);
    let mut v99 = v64.neg_mul_add(mul, v2);
    let mul = D::F32Vec::splat(d, 0.5037887256810443);
    let mut v100 = v65.mul_add(mul, v4);
    let mut v101 = v65.neg_mul_add(mul, v4);
    let mul = D::F32Vec::splat(d, 0.5074711720725553);
    let mut v102 = v66.mul_add(mul, v6);
    let mut v103 = v66.neg_mul_add(mul, v6);
    let mul = D::F32Vec::splat(d, 0.5124514794082247);
    let mut v104 = v67.mul_add(mul, v8);
    let mut v105 = v67.neg_mul_add(mul, v8);
    let mul = D::F32Vec::splat(d, 0.5187927131053328);
    let mut v106 = v68.mul_add(mul, v10);
    let mut v107 = v68.neg_mul_add(mul, v10);
    let mul = D::F32Vec::splat(d, 0.5265773151542700);
    let mut v108 = v69.mul_add(mul, v12);
    let mut v109 = v69.neg_mul_add(mul, v12);
    let mul = D::F32Vec::splat(d, 0.5359098169079920);
    let mut v110 = v70.mul_add(mul, v14);
    let mut v111 = v70.neg_mul_add(mul, v14);
    let mul = D::F32Vec::splat(d, 0.5469204379855088);
    let mut v112 = v71.mul_add(mul, v16);
    let mut v113 = v71.neg_mul_add(mul, v16);
    let mul = D::F32Vec::splat(d, 0.5597698129470802);
    let mut v114 = v72.mul_add(mul, v18);
    let mut v115 = v72.neg_mul_add(mul, v18);
    let mul = D::F32Vec::splat(d, 0.5746551840326600);
    let mut v116 = v73.mul_add(mul, v20);
    let mut v117 = v73.neg_mul_add(mul, v20);
    let mul = D::F32Vec::splat(d, 0.5918185358574165);
    let mut v118 = v74.mul_add(mul, v22);
    let mut v119 = v74.neg_mul_add(mul, v22);
    let mul = D::F32Vec::splat(d, 0.6115573478825099);
    let mut v120 = v75.mul_add(mul, v24);
    let mut v121 = v75.neg_mul_add(mul, v24);
    let mul = D::F32Vec::splat(d, 0.6342389366884031);
    let mut v122 = v76.mul_add(mul, v26);
    let mut v123 = v76.neg_mul_add(mul, v26);
    let mul = D::F32Vec::splat(d, 0.6603198078137061);
    let mut v124 = v77.mul_add(mul, v28);
    let mut v125 = v77.neg_mul_add(mul, v28);
    let mul = D::F32Vec::splat(d, 0.6903721282002123);
    let mut v126 = v78.mul_add(mul, v30);
    let mut v127 = v78.neg_mul_add(mul, v30);
    let mul = D::F32Vec::splat(d, 0.7251205223771985);
    let mut v128 = v79.mul_add(mul, v32);
    let mut v129 = v79.neg_mul_add(mul, v32);
    let mul = D::F32Vec::splat(d, 0.7654941649730891);
    let mut v130 = v80.mul_add(mul, v34);
    let mut v131 = v80.neg_mul_add(mul, v34);
    let mul = D::F32Vec::splat(d, 0.8127020908144905);
    let mut v132 = v81.mul_add(mul, v36);
    let mut v133 = v81.neg_mul_add(mul, v36);
    let mul = D::F32Vec::splat(d, 0.8683447152233481);
    let mut v134 = v82.mul_add(mul, v38);
    let mut v135 = v82.neg_mul_add(mul, v38);
    let mul = D::F32Vec::splat(d, 0.9345835970364075);
    let mut v136 = v83.mul_add(mul, v40);
    let mut v137 = v83.neg_mul_add(mul, v40);
    let mul = D::F32Vec::splat(d, 1.0144082649970547);
    let mut v138 = v84.mul_add(mul, v42);
    let mut v139 = v84.neg_mul_add(mul, v42);
    let mul = D::F32Vec::splat(d, 1.1120716205797176);
    let mut v140 = v85.mul_add(mul, v44);
    let mut v141 = v85.neg_mul_add(mul, v44);
    let mul = D::F32Vec::splat(d, 1.2338327379765710);
    let mut v142 = v86.mul_add(mul, v46);
    let mut v143 = v86.neg_mul_add(mul, v46);
    let mul = D::F32Vec::splat(d, 1.3892939586328277);
    let mut v144 = v87.mul_add(mul, v48);
    let mut v145 = v87.neg_mul_add(mul, v48);
    let mul = D::F32Vec::splat(d, 1.5939722833856311);
    let mut v146 = v88.mul_add(mul, v50);
    let mut v147 = v88.neg_mul_add(mul, v50);
    let mul = D::F32Vec::splat(d, 1.8746759800084078);
    let mut v148 = v89.mul_add(mul, v52);
    let mut v149 = v89.neg_mul_add(mul, v52);
    let mul = D::F32Vec::splat(d, 2.2820500680051619);
    let mut v150 = v90.mul_add(mul, v54);
    let mut v151 = v90.neg_mul_add(mul, v54);
    let mul = D::F32Vec::splat(d, 2.9246284281582162);
    let mut v152 = v91.mul_add(mul, v56);
    let mut v153 = v91.neg_mul_add(mul, v56);
    let mul = D::F32Vec::splat(d, 4.0846110781292477);
    let mut v154 = v92.mul_add(mul, v58);
    let mut v155 = v92.neg_mul_add(mul, v58);
    let mul = D::F32Vec::splat(d, 6.7967507116736332);
    let mut v156 = v93.mul_add(mul, v60);
    let mut v157 = v93.neg_mul_add(mul, v60);
    let mul = D::F32Vec::splat(d, 20.3738781672314531);
    let mut v158 = v94.mul_add(mul, v62);
    let mut v159 = v94.neg_mul_add(mul, v62);
    (
        v96, v98, v100, v102, v104, v106, v108, v110, v112, v114, v116, v118, v120, v122, v124,
        v126, v128, v130, v132, v134, v136, v138, v140, v142, v144, v146, v148, v150, v152, v154,
        v156, v158, v159, v157, v155, v153, v151, v149, v147, v145, v143, v141, v139, v137, v135,
        v133, v131, v129, v127, v125, v123, v121, v119, v117, v115, v113, v111, v109, v107, v105,
        v103, v101, v99, v97,
    )
}

#[inline(always)]
pub(super) fn do_idct_64<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
    stride: usize,
) {
    assert!(data.len() > 63 * stride);
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
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63,
    ) = idct_64(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63,
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
}

#[inline(always)]
pub(super) fn do_idct_64_rowblock<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
) {
    assert!(data.len() >= 64);
    const { assert!(64usize.is_multiple_of(D::F32Vec::LEN)) };
    let row_stride = 64 / D::F32Vec::LEN;
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
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63,
    ) = idct_64(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63,
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
}

#[inline(always)]
pub(super) fn do_idct_64_trh<D: SimdDescriptor>(
    d: D,
    data: &mut [<D::F32Vec as F32SimdVec>::UnderlyingArray],
) {
    let row_stride = 32 / D::F32Vec::LEN;
    assert!(data.len() > 63 * row_stride);
    const { assert!(32usize.is_multiple_of(D::F32Vec::LEN)) };
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
    let mut v32 = D::F32Vec::load_array(d, &data[row_stride * 1]);
    let mut v33 = D::F32Vec::load_array(d, &data[row_stride * 3]);
    let mut v34 = D::F32Vec::load_array(d, &data[row_stride * 5]);
    let mut v35 = D::F32Vec::load_array(d, &data[row_stride * 7]);
    let mut v36 = D::F32Vec::load_array(d, &data[row_stride * 9]);
    let mut v37 = D::F32Vec::load_array(d, &data[row_stride * 11]);
    let mut v38 = D::F32Vec::load_array(d, &data[row_stride * 13]);
    let mut v39 = D::F32Vec::load_array(d, &data[row_stride * 15]);
    let mut v40 = D::F32Vec::load_array(d, &data[row_stride * 17]);
    let mut v41 = D::F32Vec::load_array(d, &data[row_stride * 19]);
    let mut v42 = D::F32Vec::load_array(d, &data[row_stride * 21]);
    let mut v43 = D::F32Vec::load_array(d, &data[row_stride * 23]);
    let mut v44 = D::F32Vec::load_array(d, &data[row_stride * 25]);
    let mut v45 = D::F32Vec::load_array(d, &data[row_stride * 27]);
    let mut v46 = D::F32Vec::load_array(d, &data[row_stride * 29]);
    let mut v47 = D::F32Vec::load_array(d, &data[row_stride * 31]);
    let mut v48 = D::F32Vec::load_array(d, &data[row_stride * 33]);
    let mut v49 = D::F32Vec::load_array(d, &data[row_stride * 35]);
    let mut v50 = D::F32Vec::load_array(d, &data[row_stride * 37]);
    let mut v51 = D::F32Vec::load_array(d, &data[row_stride * 39]);
    let mut v52 = D::F32Vec::load_array(d, &data[row_stride * 41]);
    let mut v53 = D::F32Vec::load_array(d, &data[row_stride * 43]);
    let mut v54 = D::F32Vec::load_array(d, &data[row_stride * 45]);
    let mut v55 = D::F32Vec::load_array(d, &data[row_stride * 47]);
    let mut v56 = D::F32Vec::load_array(d, &data[row_stride * 49]);
    let mut v57 = D::F32Vec::load_array(d, &data[row_stride * 51]);
    let mut v58 = D::F32Vec::load_array(d, &data[row_stride * 53]);
    let mut v59 = D::F32Vec::load_array(d, &data[row_stride * 55]);
    let mut v60 = D::F32Vec::load_array(d, &data[row_stride * 57]);
    let mut v61 = D::F32Vec::load_array(d, &data[row_stride * 59]);
    let mut v62 = D::F32Vec::load_array(d, &data[row_stride * 61]);
    let mut v63 = D::F32Vec::load_array(d, &data[row_stride * 63]);
    (
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
        v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55,
        v56, v57, v58, v59, v60, v61, v62, v63,
    ) = idct_64(
        d, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
        v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
        v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54,
        v55, v56, v57, v58, v59, v60, v61, v62, v63,
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
}
