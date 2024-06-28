
gemm_blas_neon.o:     file format elf64-littleaarch64


Disassembly of section .text:

0000000000000000 <_Z11cacheMalloci>:
       0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
       4:	910003fd 	mov	x29, sp
       8:	b9001fe0 	str	w0, [sp, #28]
       c:	90000000 	adrp	x0, 0 <__stack_chk_guard>
      10:	f9400000 	ldr	x0, [x0]
      14:	f9400001 	ldr	x1, [x0]
      18:	f9001fe1 	str	x1, [sp, #56]
      1c:	d2800001 	mov	x1, #0x0                   	// #0
      20:	f9001bff 	str	xzr, [sp, #48]
      24:	b9801fe0 	ldrsw	x0, [sp, #28]
      28:	d37ef401 	lsl	x1, x0, #2
      2c:	9100c3e0 	add	x0, sp, #0x30
      30:	aa0103e2 	mov	x2, x1
      34:	d2800801 	mov	x1, #0x40                  	// #64
      38:	94000000 	bl	0 <posix_memalign>
      3c:	b9002fe0 	str	w0, [sp, #44]
      40:	b9402fe0 	ldr	w0, [sp, #44]
      44:	7100001f 	cmp	w0, #0x0
      48:	54000120 	b.eq	6c <_Z11cacheMalloci+0x6c>  // b.none
      4c:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
      50:	91000003 	add	x3, x0, #0x0
      54:	52800b02 	mov	w2, #0x58                  	// #88
      58:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
      5c:	91000001 	add	x1, x0, #0x0
      60:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
      64:	91000000 	add	x0, x0, #0x0
      68:	94000000 	bl	0 <__assert_fail>
      6c:	f9401be0 	ldr	x0, [sp, #48]
      70:	aa0003e1 	mov	x1, x0
      74:	90000000 	adrp	x0, 0 <__stack_chk_guard>
      78:	f9400000 	ldr	x0, [x0]
      7c:	f9401fe2 	ldr	x2, [sp, #56]
      80:	f9400003 	ldr	x3, [x0]
      84:	eb030042 	subs	x2, x2, x3
      88:	d2800003 	mov	x3, #0x0                   	// #0
      8c:	54000040 	b.eq	94 <_Z11cacheMalloci+0x94>  // b.none
      90:	94000000 	bl	0 <__stack_chk_fail>
      94:	aa0103e0 	mov	x0, x1
      98:	a8c47bfd 	ldp	x29, x30, [sp], #64
      9c:	d65f03c0 	ret

00000000000000a0 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i>:
      a0:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
      a4:	910003fd 	mov	x29, sp
      a8:	b9003fe0 	str	w0, [sp, #60]
      ac:	b9003be1 	str	w1, [sp, #56]
      b0:	b90037e2 	str	w2, [sp, #52]
      b4:	f90017e3 	str	x3, [sp, #40]
      b8:	b90033e4 	str	w4, [sp, #48]
      bc:	f90013e5 	str	x5, [sp, #32]
      c0:	b9001fe6 	str	w6, [sp, #28]
      c4:	f9000be7 	str	x7, [sp, #16]
      c8:	b9403fe0 	ldr	w0, [sp, #60]
      cc:	12000400 	and	w0, w0, #0x3
      d0:	7100001f 	cmp	w0, #0x0
      d4:	54000121 	b.ne	f8 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x58>  // b.any
      d8:	b9403be0 	ldr	w0, [sp, #56]
      dc:	12000400 	and	w0, w0, #0x3
      e0:	7100001f 	cmp	w0, #0x0
      e4:	540000a1 	b.ne	f8 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x58>  // b.any
      e8:	b94037e0 	ldr	w0, [sp, #52]
      ec:	12000400 	and	w0, w0, #0x3
      f0:	7100001f 	cmp	w0, #0x0
      f4:	54000120 	b.eq	118 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x78>  // b.none
      f8:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
      fc:	91000003 	add	x3, x0, #0x0
     100:	52800c22 	mov	w2, #0x61                  	// #97
     104:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     108:	91000001 	add	x1, x0, #0x0
     10c:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     110:	91000000 	add	x0, x0, #0x0
     114:	94000000 	bl	0 <__assert_fail>
     118:	b9403fe0 	ldr	w0, [sp, #60]
     11c:	7100001f 	cmp	w0, #0x0
     120:	540000ed 	b.le	13c <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x9c>
     124:	b9403be0 	ldr	w0, [sp, #56]
     128:	7100001f 	cmp	w0, #0x0
     12c:	5400008d 	b.le	13c <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x9c>
     130:	b94037e0 	ldr	w0, [sp, #52]
     134:	7100001f 	cmp	w0, #0x0
     138:	5400012c 	b.gt	15c <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0xbc>
     13c:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     140:	91000003 	add	x3, x0, #0x0
     144:	52800c42 	mov	w2, #0x62                  	// #98
     148:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     14c:	91000001 	add	x1, x0, #0x0
     150:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     154:	91000000 	add	x0, x0, #0x0
     158:	94000000 	bl	0 <__assert_fail>
     15c:	b9403fe1 	ldr	w1, [sp, #60]
     160:	b94037e0 	ldr	w0, [sp, #52]
     164:	1b007c20 	mul	w0, w1, w0
     168:	94000000 	bl	0 <_Z11cacheMalloci>
     16c:	f90033e0 	str	x0, [sp, #96]
     170:	b94037e1 	ldr	w1, [sp, #52]
     174:	b9403be0 	ldr	w0, [sp, #56]
     178:	1b007c20 	mul	w0, w1, w0
     17c:	94000000 	bl	0 <_Z11cacheMalloci>
     180:	f90037e0 	str	x0, [sp, #104]
     184:	b9004fff 	str	wzr, [sp, #76]
     188:	b9404fe1 	ldr	w1, [sp, #76]
     18c:	b9403fe0 	ldr	w0, [sp, #60]
     190:	6b00003f 	cmp	w1, w0
     194:	54001baa 	b.ge	508 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x468>  // b.tcont
     198:	b9403fe1 	ldr	w1, [sp, #60]
     19c:	b9404fe0 	ldr	w0, [sp, #76]
     1a0:	4b000020 	sub	w0, w1, w0
     1a4:	b90043e0 	str	w0, [sp, #64]
     1a8:	b94043e0 	ldr	w0, [sp, #64]
     1ac:	7110001f 	cmp	w0, #0x400
     1b0:	5400006d 	b.le	1bc <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x11c>
     1b4:	52808000 	mov	w0, #0x400                 	// #1024
     1b8:	b90043e0 	str	w0, [sp, #64]
     1bc:	b90057ff 	str	wzr, [sp, #84]
     1c0:	b94057e1 	ldr	w1, [sp, #84]
     1c4:	b94037e0 	ldr	w0, [sp, #52]
     1c8:	6b00003f 	cmp	w1, w0
     1cc:	5400196a 	b.ge	4f8 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x458>  // b.tcont
     1d0:	b94037e1 	ldr	w1, [sp, #52]
     1d4:	b94057e0 	ldr	w0, [sp, #84]
     1d8:	4b000020 	sub	w0, w1, w0
     1dc:	b9004be0 	str	w0, [sp, #72]
     1e0:	b9404be0 	ldr	w0, [sp, #72]
     1e4:	7107fc1f 	cmp	w0, #0x1ff
     1e8:	5400008d 	b.le	1f8 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x158>
     1ec:	52802000 	mov	w0, #0x100                 	// #256
     1f0:	b9004be0 	str	w0, [sp, #72]
     1f4:	1400000b 	b	220 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x180>
     1f8:	b9404be0 	ldr	w0, [sp, #72]
     1fc:	7104001f 	cmp	w0, #0x100
     200:	5400010d 	b.le	220 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x180>
     204:	b9404be0 	ldr	w0, [sp, #72]
     208:	531f7c01 	lsr	w1, w0, #31
     20c:	0b000020 	add	w0, w1, w0
     210:	13017c00 	asr	w0, w0, #1
     214:	11000c00 	add	w0, w0, #0x3
     218:	121e7400 	and	w0, w0, #0xfffffffc
     21c:	b9004be0 	str	w0, [sp, #72]
     220:	b9403be0 	ldr	w0, [sp, #56]
     224:	b90047e0 	str	w0, [sp, #68]
     228:	b94047e0 	ldr	w0, [sp, #68]
     22c:	7107fc1f 	cmp	w0, #0x1ff
     230:	5400008d 	b.le	240 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x1a0>
     234:	52802000 	mov	w0, #0x100                 	// #256
     238:	b90047e0 	str	w0, [sp, #68]
     23c:	1400000b 	b	268 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x1c8>
     240:	b94047e0 	ldr	w0, [sp, #68]
     244:	7104001f 	cmp	w0, #0x100
     248:	5400010d 	b.le	268 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x1c8>
     24c:	b94047e0 	ldr	w0, [sp, #68]
     250:	531f7c01 	lsr	w1, w0, #31
     254:	0b000020 	add	w0, w1, w0
     258:	13017c00 	asr	w0, w0, #1
     25c:	11000c00 	add	w0, w0, #0x3
     260:	121e7400 	and	w0, w0, #0xfffffffc
     264:	b90047e0 	str	w0, [sp, #68]
     268:	b94057e1 	ldr	w1, [sp, #84]
     26c:	b9401fe0 	ldr	w0, [sp, #28]
     270:	1b007c20 	mul	w0, w1, w0
     274:	93407c00 	sxtw	x0, w0
     278:	d37ef400 	lsl	x0, x0, #2
     27c:	f94013e1 	ldr	x1, [sp, #32]
     280:	8b000020 	add	x0, x1, x0
     284:	f94037e4 	ldr	x4, [sp, #104]
     288:	b9401fe3 	ldr	w3, [sp, #28]
     28c:	aa0003e2 	mov	x2, x0
     290:	b94047e1 	ldr	w1, [sp, #68]
     294:	b9404be0 	ldr	w0, [sp, #72]
     298:	94000000 	bl	874 <_Z5packBiiPfiS_>
     29c:	b9404fe0 	ldr	w0, [sp, #76]
     2a0:	b9005fe0 	str	w0, [sp, #92]
     2a4:	b9404fe1 	ldr	w1, [sp, #76]
     2a8:	b94043e0 	ldr	w0, [sp, #64]
     2ac:	0b000020 	add	w0, w1, w0
     2b0:	b9405fe1 	ldr	w1, [sp, #92]
     2b4:	6b00003f 	cmp	w1, w0
     2b8:	5400098a 	b.ge	3e8 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x348>  // b.tcont
     2bc:	b9404fe1 	ldr	w1, [sp, #76]
     2c0:	b94043e0 	ldr	w0, [sp, #64]
     2c4:	0b000021 	add	w1, w1, w0
     2c8:	b9405fe0 	ldr	w0, [sp, #92]
     2cc:	4b000020 	sub	w0, w1, w0
     2d0:	b9005be0 	str	w0, [sp, #88]
     2d4:	b9405be0 	ldr	w0, [sp, #88]
     2d8:	71002c1f 	cmp	w0, #0xb
     2dc:	5400008d 	b.le	2ec <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x24c>
     2e0:	52800180 	mov	w0, #0xc                   	// #12
     2e4:	b9005be0 	str	w0, [sp, #88]
     2e8:	1400000c 	b	318 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x278>
     2ec:	b9405be0 	ldr	w0, [sp, #88]
     2f0:	71001c1f 	cmp	w0, #0x7
     2f4:	5400008d 	b.le	304 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x264>
     2f8:	52800100 	mov	w0, #0x8                   	// #8
     2fc:	b9005be0 	str	w0, [sp, #88]
     300:	14000006 	b	318 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x278>
     304:	b9405be0 	ldr	w0, [sp, #88]
     308:	71000c1f 	cmp	w0, #0x3
     30c:	5400006d 	b.le	318 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x278>
     310:	52800080 	mov	w0, #0x4                   	// #4
     314:	b9005be0 	str	w0, [sp, #88]
     318:	b9405fe1 	ldr	w1, [sp, #92]
     31c:	b94033e0 	ldr	w0, [sp, #48]
     320:	1b007c20 	mul	w0, w1, w0
     324:	93407c01 	sxtw	x1, w0
     328:	b98057e0 	ldrsw	x0, [sp, #84]
     32c:	8b000020 	add	x0, x1, x0
     330:	d37ef400 	lsl	x0, x0, #2
     334:	f94017e1 	ldr	x1, [sp, #40]
     338:	8b000022 	add	x2, x1, x0
     33c:	b9405fe1 	ldr	w1, [sp, #92]
     340:	b9404fe0 	ldr	w0, [sp, #76]
     344:	4b000021 	sub	w1, w1, w0
     348:	b9404be0 	ldr	w0, [sp, #72]
     34c:	1b007c20 	mul	w0, w1, w0
     350:	93407c00 	sxtw	x0, w0
     354:	d37ef400 	lsl	x0, x0, #2
     358:	f94033e1 	ldr	x1, [sp, #96]
     35c:	8b000020 	add	x0, x1, x0
     360:	aa0003e4 	mov	x4, x0
     364:	b94033e3 	ldr	w3, [sp, #48]
     368:	b9404be1 	ldr	w1, [sp, #72]
     36c:	b9405be0 	ldr	w0, [sp, #88]
     370:	94000000 	bl	524 <_Z5packAiiPfiS_>
     374:	b9405fe1 	ldr	w1, [sp, #92]
     378:	b9404fe0 	ldr	w0, [sp, #76]
     37c:	4b000021 	sub	w1, w1, w0
     380:	b9404be0 	ldr	w0, [sp, #72]
     384:	1b007c20 	mul	w0, w1, w0
     388:	93407c00 	sxtw	x0, w0
     38c:	d37ef400 	lsl	x0, x0, #2
     390:	f94033e1 	ldr	x1, [sp, #96]
     394:	8b000022 	add	x2, x1, x0
     398:	b9405fe1 	ldr	w1, [sp, #92]
     39c:	b94073e0 	ldr	w0, [sp, #112]
     3a0:	1b007c20 	mul	w0, w1, w0
     3a4:	93407c00 	sxtw	x0, w0
     3a8:	d37ef400 	lsl	x0, x0, #2
     3ac:	f9400be1 	ldr	x1, [sp, #16]
     3b0:	8b000020 	add	x0, x1, x0
     3b4:	b94073e6 	ldr	w6, [sp, #112]
     3b8:	aa0003e5 	mov	x5, x0
     3bc:	f94037e4 	ldr	x4, [sp, #104]
     3c0:	aa0203e3 	mov	x3, x2
     3c4:	b9404be2 	ldr	w2, [sp, #72]
     3c8:	b94047e1 	ldr	w1, [sp, #68]
     3cc:	b9405be0 	ldr	w0, [sp, #88]
     3d0:	94000000 	bl	b8c <_Z10kernel_4x4iiiPfS_S_i>
     3d4:	b9405fe1 	ldr	w1, [sp, #92]
     3d8:	b9405be0 	ldr	w0, [sp, #88]
     3dc:	0b000020 	add	w0, w1, w0
     3e0:	b9005fe0 	str	w0, [sp, #92]
     3e4:	17ffffb0 	b	2a4 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x204>
     3e8:	b94047e0 	ldr	w0, [sp, #68]
     3ec:	b90053e0 	str	w0, [sp, #80]
     3f0:	b94053e1 	ldr	w1, [sp, #80]
     3f4:	b9403be0 	ldr	w0, [sp, #56]
     3f8:	6b00003f 	cmp	w1, w0
     3fc:	5400074a 	b.ge	4e4 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x444>  // b.tcont
     400:	b9403be1 	ldr	w1, [sp, #56]
     404:	b94053e0 	ldr	w0, [sp, #80]
     408:	4b000020 	sub	w0, w1, w0
     40c:	b90047e0 	str	w0, [sp, #68]
     410:	b94047e0 	ldr	w0, [sp, #68]
     414:	7107fc1f 	cmp	w0, #0x1ff
     418:	5400008d 	b.le	428 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x388>
     41c:	52802000 	mov	w0, #0x100                 	// #256
     420:	b90047e0 	str	w0, [sp, #68]
     424:	1400000b 	b	450 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x3b0>
     428:	b94047e0 	ldr	w0, [sp, #68]
     42c:	7104001f 	cmp	w0, #0x100
     430:	5400010d 	b.le	450 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x3b0>
     434:	b94047e0 	ldr	w0, [sp, #68]
     438:	531f7c01 	lsr	w1, w0, #31
     43c:	0b000020 	add	w0, w1, w0
     440:	13017c00 	asr	w0, w0, #1
     444:	11000c00 	add	w0, w0, #0x3
     448:	121e7400 	and	w0, w0, #0xfffffffc
     44c:	b90047e0 	str	w0, [sp, #68]
     450:	b94057e1 	ldr	w1, [sp, #84]
     454:	b9401fe0 	ldr	w0, [sp, #28]
     458:	1b007c20 	mul	w0, w1, w0
     45c:	93407c01 	sxtw	x1, w0
     460:	b98053e0 	ldrsw	x0, [sp, #80]
     464:	8b000020 	add	x0, x1, x0
     468:	d37ef400 	lsl	x0, x0, #2
     46c:	f94013e1 	ldr	x1, [sp, #32]
     470:	8b000020 	add	x0, x1, x0
     474:	f94037e4 	ldr	x4, [sp, #104]
     478:	b9401fe3 	ldr	w3, [sp, #28]
     47c:	aa0003e2 	mov	x2, x0
     480:	b94047e1 	ldr	w1, [sp, #68]
     484:	b9404be0 	ldr	w0, [sp, #72]
     488:	94000000 	bl	874 <_Z5packBiiPfiS_>
     48c:	b9404fe1 	ldr	w1, [sp, #76]
     490:	b94073e0 	ldr	w0, [sp, #112]
     494:	1b007c20 	mul	w0, w1, w0
     498:	93407c01 	sxtw	x1, w0
     49c:	b98053e0 	ldrsw	x0, [sp, #80]
     4a0:	8b000020 	add	x0, x1, x0
     4a4:	d37ef400 	lsl	x0, x0, #2
     4a8:	f9400be1 	ldr	x1, [sp, #16]
     4ac:	8b000020 	add	x0, x1, x0
     4b0:	b94073e6 	ldr	w6, [sp, #112]
     4b4:	aa0003e5 	mov	x5, x0
     4b8:	f94037e4 	ldr	x4, [sp, #104]
     4bc:	f94033e3 	ldr	x3, [sp, #96]
     4c0:	b9404be2 	ldr	w2, [sp, #72]
     4c4:	b94047e1 	ldr	w1, [sp, #68]
     4c8:	b94043e0 	ldr	w0, [sp, #64]
     4cc:	94000000 	bl	b8c <_Z10kernel_4x4iiiPfS_S_i>
     4d0:	b94053e1 	ldr	w1, [sp, #80]
     4d4:	b94047e0 	ldr	w0, [sp, #68]
     4d8:	0b000020 	add	w0, w1, w0
     4dc:	b90053e0 	str	w0, [sp, #80]
     4e0:	17ffffc4 	b	3f0 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x350>
     4e4:	b94057e1 	ldr	w1, [sp, #84]
     4e8:	b9404be0 	ldr	w0, [sp, #72]
     4ec:	0b000020 	add	w0, w1, w0
     4f0:	b90057e0 	str	w0, [sp, #84]
     4f4:	17ffff33 	b	1c0 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0x120>
     4f8:	b9404fe0 	ldr	w0, [sp, #76]
     4fc:	11100000 	add	w0, w0, #0x400
     500:	b9004fe0 	str	w0, [sp, #76]
     504:	17ffff21 	b	188 <_Z23gemm_4x4_like_blas_neoniiiPfiS_iS_i+0xe8>
     508:	f94037e0 	ldr	x0, [sp, #104]
     50c:	94000000 	bl	0 <free>
     510:	f94033e0 	ldr	x0, [sp, #96]
     514:	94000000 	bl	0 <free>
     518:	d503201f 	nop
     51c:	a8c77bfd 	ldp	x29, x30, [sp], #112
     520:	d65f03c0 	ret

0000000000000524 <_Z5packAiiPfiS_>:
     524:	a9b57bfd 	stp	x29, x30, [sp, #-176]!
     528:	910003fd 	mov	x29, sp
     52c:	b9002fe0 	str	w0, [sp, #44]
     530:	b9002be1 	str	w1, [sp, #40]
     534:	f90013e2 	str	x2, [sp, #32]
     538:	b9001fe3 	str	w3, [sp, #28]
     53c:	f9000be4 	str	x4, [sp, #16]
     540:	b9402fe0 	ldr	w0, [sp, #44]
     544:	7100001f 	cmp	w0, #0x0
     548:	5400018d 	b.le	578 <_Z5packAiiPfiS_+0x54>
     54c:	b9402be0 	ldr	w0, [sp, #40]
     550:	7100001f 	cmp	w0, #0x0
     554:	5400012d 	b.le	578 <_Z5packAiiPfiS_+0x54>
     558:	b9402fe0 	ldr	w0, [sp, #44]
     55c:	12000400 	and	w0, w0, #0x3
     560:	7100001f 	cmp	w0, #0x0
     564:	540000a1 	b.ne	578 <_Z5packAiiPfiS_+0x54>  // b.any
     568:	b9402be0 	ldr	w0, [sp, #40]
     56c:	12000400 	and	w0, w0, #0x3
     570:	7100001f 	cmp	w0, #0x0
     574:	54000120 	b.eq	598 <_Z5packAiiPfiS_+0x74>  // b.none
     578:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     57c:	91000003 	add	x3, x0, #0x0
     580:	528015c2 	mov	w2, #0xae                  	// #174
     584:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     588:	91000001 	add	x1, x0, #0x0
     58c:	90000000 	adrp	x0, 0 <_Z11cacheMalloci>
     590:	91000000 	add	x0, x0, #0x0
     594:	94000000 	bl	0 <__assert_fail>
     598:	f94013e0 	ldr	x0, [sp, #32]
     59c:	f90043e0 	str	x0, [sp, #128]
     5a0:	f9400be0 	ldr	x0, [sp, #16]
     5a4:	f90047e0 	str	x0, [sp, #136]
     5a8:	b9003bff 	str	wzr, [sp, #56]
     5ac:	b9403be0 	ldr	w0, [sp, #56]
     5b0:	11000c00 	add	w0, w0, #0x3
     5b4:	b9402fe1 	ldr	w1, [sp, #44]
     5b8:	6b00003f 	cmp	w1, w0
     5bc:	5400156d 	b.le	868 <_Z5packAiiPfiS_+0x344>
     5c0:	f94043e0 	ldr	x0, [sp, #128]
     5c4:	f9004be0 	str	x0, [sp, #144]
     5c8:	b9801fe0 	ldrsw	x0, [sp, #28]
     5cc:	d37ef400 	lsl	x0, x0, #2
     5d0:	f94043e1 	ldr	x1, [sp, #128]
     5d4:	8b000020 	add	x0, x1, x0
     5d8:	f9004fe0 	str	x0, [sp, #152]
     5dc:	b9401fe0 	ldr	w0, [sp, #28]
     5e0:	531f7800 	lsl	w0, w0, #1
     5e4:	93407c00 	sxtw	x0, w0
     5e8:	d37ef400 	lsl	x0, x0, #2
     5ec:	f94043e1 	ldr	x1, [sp, #128]
     5f0:	8b000020 	add	x0, x1, x0
     5f4:	f90053e0 	str	x0, [sp, #160]
     5f8:	b9401fe1 	ldr	w1, [sp, #28]
     5fc:	2a0103e0 	mov	w0, w1
     600:	531f7800 	lsl	w0, w0, #1
     604:	0b010000 	add	w0, w0, w1
     608:	93407c00 	sxtw	x0, w0
     60c:	d37ef400 	lsl	x0, x0, #2
     610:	f94043e1 	ldr	x1, [sp, #128]
     614:	8b000020 	add	x0, x1, x0
     618:	f90057e0 	str	x0, [sp, #168]
     61c:	b9003fff 	str	wzr, [sp, #60]
     620:	b9403fe0 	ldr	w0, [sp, #60]
     624:	11000c00 	add	w0, w0, #0x3
     628:	b9402be1 	ldr	w1, [sp, #40]
     62c:	6b00003f 	cmp	w1, w0
     630:	5400106d 	b.le	83c <_Z5packAiiPfiS_+0x318>
     634:	f9404be0 	ldr	x0, [sp, #144]
     638:	bd400000 	ldr	s0, [x0]
     63c:	bd0043e0 	str	s0, [sp, #64]
     640:	f9404fe0 	ldr	x0, [sp, #152]
     644:	bd400000 	ldr	s0, [x0]
     648:	bd0047e0 	str	s0, [sp, #68]
     64c:	f94053e0 	ldr	x0, [sp, #160]
     650:	bd400000 	ldr	s0, [x0]
     654:	bd004be0 	str	s0, [sp, #72]
     658:	f94057e0 	ldr	x0, [sp, #168]
     65c:	bd400000 	ldr	s0, [x0]
     660:	bd004fe0 	str	s0, [sp, #76]
     664:	f9404be0 	ldr	x0, [sp, #144]
     668:	bd400400 	ldr	s0, [x0, #4]
     66c:	bd0053e0 	str	s0, [sp, #80]
     670:	f9404fe0 	ldr	x0, [sp, #152]
     674:	bd400400 	ldr	s0, [x0, #4]
     678:	bd0057e0 	str	s0, [sp, #84]
     67c:	f94053e0 	ldr	x0, [sp, #160]
     680:	bd400400 	ldr	s0, [x0, #4]
     684:	bd005be0 	str	s0, [sp, #88]
     688:	f94057e0 	ldr	x0, [sp, #168]
     68c:	bd400400 	ldr	s0, [x0, #4]
     690:	bd005fe0 	str	s0, [sp, #92]
     694:	f9404be0 	ldr	x0, [sp, #144]
     698:	bd400800 	ldr	s0, [x0, #8]
     69c:	bd0063e0 	str	s0, [sp, #96]
     6a0:	f9404fe0 	ldr	x0, [sp, #152]
     6a4:	bd400800 	ldr	s0, [x0, #8]
     6a8:	bd0067e0 	str	s0, [sp, #100]
     6ac:	f94053e0 	ldr	x0, [sp, #160]
     6b0:	bd400800 	ldr	s0, [x0, #8]
     6b4:	bd006be0 	str	s0, [sp, #104]
     6b8:	f94057e0 	ldr	x0, [sp, #168]
     6bc:	bd400800 	ldr	s0, [x0, #8]
     6c0:	bd006fe0 	str	s0, [sp, #108]
     6c4:	f9404be0 	ldr	x0, [sp, #144]
     6c8:	bd400c00 	ldr	s0, [x0, #12]
     6cc:	bd0073e0 	str	s0, [sp, #112]
     6d0:	f9404fe0 	ldr	x0, [sp, #152]
     6d4:	bd400c00 	ldr	s0, [x0, #12]
     6d8:	bd0077e0 	str	s0, [sp, #116]
     6dc:	f94053e0 	ldr	x0, [sp, #160]
     6e0:	bd400c00 	ldr	s0, [x0, #12]
     6e4:	bd007be0 	str	s0, [sp, #120]
     6e8:	f94057e0 	ldr	x0, [sp, #168]
     6ec:	bd400c00 	ldr	s0, [x0, #12]
     6f0:	bd007fe0 	str	s0, [sp, #124]
     6f4:	f94047e0 	ldr	x0, [sp, #136]
     6f8:	bd4043e0 	ldr	s0, [sp, #64]
     6fc:	bd000000 	str	s0, [x0]
     700:	f94047e0 	ldr	x0, [sp, #136]
     704:	91001000 	add	x0, x0, #0x4
     708:	bd4047e0 	ldr	s0, [sp, #68]
     70c:	bd000000 	str	s0, [x0]
     710:	f94047e0 	ldr	x0, [sp, #136]
     714:	91002000 	add	x0, x0, #0x8
     718:	bd404be0 	ldr	s0, [sp, #72]
     71c:	bd000000 	str	s0, [x0]
     720:	f94047e0 	ldr	x0, [sp, #136]
     724:	91003000 	add	x0, x0, #0xc
     728:	bd404fe0 	ldr	s0, [sp, #76]
     72c:	bd000000 	str	s0, [x0]
     730:	f94047e0 	ldr	x0, [sp, #136]
     734:	91004000 	add	x0, x0, #0x10
     738:	bd4053e0 	ldr	s0, [sp, #80]
     73c:	bd000000 	str	s0, [x0]
     740:	f94047e0 	ldr	x0, [sp, #136]
     744:	91005000 	add	x0, x0, #0x14
     748:	bd4057e0 	ldr	s0, [sp, #84]
     74c:	bd000000 	str	s0, [x0]
     750:	f94047e0 	ldr	x0, [sp, #136]
     754:	91006000 	add	x0, x0, #0x18
     758:	bd405be0 	ldr	s0, [sp, #88]
     75c:	bd000000 	str	s0, [x0]
     760:	f94047e0 	ldr	x0, [sp, #136]
     764:	91007000 	add	x0, x0, #0x1c
     768:	bd405fe0 	ldr	s0, [sp, #92]
     76c:	bd000000 	str	s0, [x0]
     770:	f94047e0 	ldr	x0, [sp, #136]
     774:	91008000 	add	x0, x0, #0x20
     778:	bd4063e0 	ldr	s0, [sp, #96]
     77c:	bd000000 	str	s0, [x0]
     780:	f94047e0 	ldr	x0, [sp, #136]
     784:	91009000 	add	x0, x0, #0x24
     788:	bd4067e0 	ldr	s0, [sp, #100]
     78c:	bd000000 	str	s0, [x0]
     790:	f94047e0 	ldr	x0, [sp, #136]
     794:	9100a000 	add	x0, x0, #0x28
     798:	bd406be0 	ldr	s0, [sp, #104]
     79c:	bd000000 	str	s0, [x0]
     7a0:	f94047e0 	ldr	x0, [sp, #136]
     7a4:	9100b000 	add	x0, x0, #0x2c
     7a8:	bd406fe0 	ldr	s0, [sp, #108]
     7ac:	bd000000 	str	s0, [x0]
     7b0:	f94047e0 	ldr	x0, [sp, #136]
     7b4:	9100c000 	add	x0, x0, #0x30
     7b8:	bd4073e0 	ldr	s0, [sp, #112]
     7bc:	bd000000 	str	s0, [x0]
     7c0:	f94047e0 	ldr	x0, [sp, #136]
     7c4:	9100d000 	add	x0, x0, #0x34
     7c8:	bd4077e0 	ldr	s0, [sp, #116]
     7cc:	bd000000 	str	s0, [x0]
     7d0:	f94047e0 	ldr	x0, [sp, #136]
     7d4:	9100e000 	add	x0, x0, #0x38
     7d8:	bd407be0 	ldr	s0, [sp, #120]
     7dc:	bd000000 	str	s0, [x0]
     7e0:	f94047e0 	ldr	x0, [sp, #136]
     7e4:	9100f000 	add	x0, x0, #0x3c
     7e8:	bd407fe0 	ldr	s0, [sp, #124]
     7ec:	bd000000 	str	s0, [x0]
     7f0:	f9404be0 	ldr	x0, [sp, #144]
     7f4:	91004000 	add	x0, x0, #0x10
     7f8:	f9004be0 	str	x0, [sp, #144]
     7fc:	f9404fe0 	ldr	x0, [sp, #152]
     800:	91004000 	add	x0, x0, #0x10
     804:	f9004fe0 	str	x0, [sp, #152]
     808:	f94053e0 	ldr	x0, [sp, #160]
     80c:	91004000 	add	x0, x0, #0x10
     810:	f90053e0 	str	x0, [sp, #160]
     814:	f94057e0 	ldr	x0, [sp, #168]
     818:	91004000 	add	x0, x0, #0x10
     81c:	f90057e0 	str	x0, [sp, #168]
     820:	f94047e0 	ldr	x0, [sp, #136]
     824:	91010000 	add	x0, x0, #0x40
     828:	f90047e0 	str	x0, [sp, #136]
     82c:	b9403fe0 	ldr	w0, [sp, #60]
     830:	11001000 	add	w0, w0, #0x4
     834:	b9003fe0 	str	w0, [sp, #60]
     838:	17ffff7a 	b	620 <_Z5packAiiPfiS_+0xfc>
     83c:	b9401fe0 	ldr	w0, [sp, #28]
     840:	531e7400 	lsl	w0, w0, #2
     844:	93407c00 	sxtw	x0, w0
     848:	d37ef400 	lsl	x0, x0, #2
     84c:	f94043e1 	ldr	x1, [sp, #128]
     850:	8b000020 	add	x0, x1, x0
     854:	f90043e0 	str	x0, [sp, #128]
     858:	b9403be0 	ldr	w0, [sp, #56]
     85c:	11001000 	add	w0, w0, #0x4
     860:	b9003be0 	str	w0, [sp, #56]
     864:	17ffff52 	b	5ac <_Z5packAiiPfiS_+0x88>
     868:	d503201f 	nop
     86c:	a8cb7bfd 	ldp	x29, x30, [sp], #176
     870:	d65f03c0 	ret

0000000000000874 <_Z5packBiiPfiS_>:
     874:	d10283ff 	sub	sp, sp, #0xa0
     878:	b9001fe0 	str	w0, [sp, #28]
     87c:	b9001be1 	str	w1, [sp, #24]
     880:	f9000be2 	str	x2, [sp, #16]
     884:	b9000fe3 	str	w3, [sp, #12]
     888:	f90003e4 	str	x4, [sp]
     88c:	f9400be0 	ldr	x0, [sp, #16]
     890:	f9003be0 	str	x0, [sp, #112]
     894:	b9002bff 	str	wzr, [sp, #40]
     898:	b9402be0 	ldr	w0, [sp, #40]
     89c:	11000c00 	add	w0, w0, #0x3
     8a0:	b9401fe1 	ldr	w1, [sp, #28]
     8a4:	6b00003f 	cmp	w1, w0
     8a8:	540016cd 	b.le	b80 <_Z5packBiiPfiS_+0x30c>
     8ac:	b9402be0 	ldr	w0, [sp, #40]
     8b0:	531e7400 	lsl	w0, w0, #2
     8b4:	93407c00 	sxtw	x0, w0
     8b8:	d37ef400 	lsl	x0, x0, #2
     8bc:	f94003e1 	ldr	x1, [sp]
     8c0:	8b000020 	add	x0, x1, x0
     8c4:	f9003fe0 	str	x0, [sp, #120]
     8c8:	f9403be0 	ldr	x0, [sp, #112]
     8cc:	f90043e0 	str	x0, [sp, #128]
     8d0:	b9800fe0 	ldrsw	x0, [sp, #12]
     8d4:	d37ef400 	lsl	x0, x0, #2
     8d8:	f9403be1 	ldr	x1, [sp, #112]
     8dc:	8b000020 	add	x0, x1, x0
     8e0:	f90047e0 	str	x0, [sp, #136]
     8e4:	b9400fe0 	ldr	w0, [sp, #12]
     8e8:	531f7800 	lsl	w0, w0, #1
     8ec:	93407c00 	sxtw	x0, w0
     8f0:	d37ef400 	lsl	x0, x0, #2
     8f4:	f9403be1 	ldr	x1, [sp, #112]
     8f8:	8b000020 	add	x0, x1, x0
     8fc:	f9004be0 	str	x0, [sp, #144]
     900:	b9400fe1 	ldr	w1, [sp, #12]
     904:	2a0103e0 	mov	w0, w1
     908:	531f7800 	lsl	w0, w0, #1
     90c:	0b010000 	add	w0, w0, w1
     910:	93407c00 	sxtw	x0, w0
     914:	d37ef400 	lsl	x0, x0, #2
     918:	f9403be1 	ldr	x1, [sp, #112]
     91c:	8b000020 	add	x0, x1, x0
     920:	f9004fe0 	str	x0, [sp, #152]
     924:	b9002fff 	str	wzr, [sp, #44]
     928:	b9402fe0 	ldr	w0, [sp, #44]
     92c:	11000c00 	add	w0, w0, #0x3
     930:	b9401be1 	ldr	w1, [sp, #24]
     934:	6b00003f 	cmp	w1, w0
     938:	540010ed 	b.le	b54 <_Z5packBiiPfiS_+0x2e0>
     93c:	f94043e0 	ldr	x0, [sp, #128]
     940:	bd400000 	ldr	s0, [x0]
     944:	bd0033e0 	str	s0, [sp, #48]
     948:	f94043e0 	ldr	x0, [sp, #128]
     94c:	bd400400 	ldr	s0, [x0, #4]
     950:	bd0037e0 	str	s0, [sp, #52]
     954:	f94043e0 	ldr	x0, [sp, #128]
     958:	bd400800 	ldr	s0, [x0, #8]
     95c:	bd003be0 	str	s0, [sp, #56]
     960:	f94043e0 	ldr	x0, [sp, #128]
     964:	bd400c00 	ldr	s0, [x0, #12]
     968:	bd003fe0 	str	s0, [sp, #60]
     96c:	f94047e0 	ldr	x0, [sp, #136]
     970:	bd400000 	ldr	s0, [x0]
     974:	bd0043e0 	str	s0, [sp, #64]
     978:	f94047e0 	ldr	x0, [sp, #136]
     97c:	bd400400 	ldr	s0, [x0, #4]
     980:	bd0047e0 	str	s0, [sp, #68]
     984:	f94047e0 	ldr	x0, [sp, #136]
     988:	bd400800 	ldr	s0, [x0, #8]
     98c:	bd004be0 	str	s0, [sp, #72]
     990:	f94047e0 	ldr	x0, [sp, #136]
     994:	bd400c00 	ldr	s0, [x0, #12]
     998:	bd004fe0 	str	s0, [sp, #76]
     99c:	f9404be0 	ldr	x0, [sp, #144]
     9a0:	bd400000 	ldr	s0, [x0]
     9a4:	bd0053e0 	str	s0, [sp, #80]
     9a8:	f9404be0 	ldr	x0, [sp, #144]
     9ac:	bd400400 	ldr	s0, [x0, #4]
     9b0:	bd0057e0 	str	s0, [sp, #84]
     9b4:	f9404be0 	ldr	x0, [sp, #144]
     9b8:	bd400800 	ldr	s0, [x0, #8]
     9bc:	bd005be0 	str	s0, [sp, #88]
     9c0:	f9404be0 	ldr	x0, [sp, #144]
     9c4:	bd400c00 	ldr	s0, [x0, #12]
     9c8:	bd005fe0 	str	s0, [sp, #92]
     9cc:	f9404fe0 	ldr	x0, [sp, #152]
     9d0:	bd400000 	ldr	s0, [x0]
     9d4:	bd0063e0 	str	s0, [sp, #96]
     9d8:	f9404fe0 	ldr	x0, [sp, #152]
     9dc:	bd400400 	ldr	s0, [x0, #4]
     9e0:	bd0067e0 	str	s0, [sp, #100]
     9e4:	f9404fe0 	ldr	x0, [sp, #152]
     9e8:	bd400800 	ldr	s0, [x0, #8]
     9ec:	bd006be0 	str	s0, [sp, #104]
     9f0:	f9404fe0 	ldr	x0, [sp, #152]
     9f4:	bd400c00 	ldr	s0, [x0, #12]
     9f8:	bd006fe0 	str	s0, [sp, #108]
     9fc:	f9403fe0 	ldr	x0, [sp, #120]
     a00:	bd4033e0 	ldr	s0, [sp, #48]
     a04:	bd000000 	str	s0, [x0]
     a08:	f9403fe0 	ldr	x0, [sp, #120]
     a0c:	91001000 	add	x0, x0, #0x4
     a10:	bd4037e0 	ldr	s0, [sp, #52]
     a14:	bd000000 	str	s0, [x0]
     a18:	f9403fe0 	ldr	x0, [sp, #120]
     a1c:	91002000 	add	x0, x0, #0x8
     a20:	bd403be0 	ldr	s0, [sp, #56]
     a24:	bd000000 	str	s0, [x0]
     a28:	f9403fe0 	ldr	x0, [sp, #120]
     a2c:	91003000 	add	x0, x0, #0xc
     a30:	bd403fe0 	ldr	s0, [sp, #60]
     a34:	bd000000 	str	s0, [x0]
     a38:	f9403fe0 	ldr	x0, [sp, #120]
     a3c:	91004000 	add	x0, x0, #0x10
     a40:	bd4043e0 	ldr	s0, [sp, #64]
     a44:	bd000000 	str	s0, [x0]
     a48:	f9403fe0 	ldr	x0, [sp, #120]
     a4c:	91005000 	add	x0, x0, #0x14
     a50:	bd4047e0 	ldr	s0, [sp, #68]
     a54:	bd000000 	str	s0, [x0]
     a58:	f9403fe0 	ldr	x0, [sp, #120]
     a5c:	91006000 	add	x0, x0, #0x18
     a60:	bd404be0 	ldr	s0, [sp, #72]
     a64:	bd000000 	str	s0, [x0]
     a68:	f9403fe0 	ldr	x0, [sp, #120]
     a6c:	91007000 	add	x0, x0, #0x1c
     a70:	bd404fe0 	ldr	s0, [sp, #76]
     a74:	bd000000 	str	s0, [x0]
     a78:	f9403fe0 	ldr	x0, [sp, #120]
     a7c:	91008000 	add	x0, x0, #0x20
     a80:	bd4053e0 	ldr	s0, [sp, #80]
     a84:	bd000000 	str	s0, [x0]
     a88:	f9403fe0 	ldr	x0, [sp, #120]
     a8c:	91009000 	add	x0, x0, #0x24
     a90:	bd4057e0 	ldr	s0, [sp, #84]
     a94:	bd000000 	str	s0, [x0]
     a98:	f9403fe0 	ldr	x0, [sp, #120]
     a9c:	9100a000 	add	x0, x0, #0x28
     aa0:	bd405be0 	ldr	s0, [sp, #88]
     aa4:	bd000000 	str	s0, [x0]
     aa8:	f9403fe0 	ldr	x0, [sp, #120]
     aac:	9100b000 	add	x0, x0, #0x2c
     ab0:	bd405fe0 	ldr	s0, [sp, #92]
     ab4:	bd000000 	str	s0, [x0]
     ab8:	f9403fe0 	ldr	x0, [sp, #120]
     abc:	9100c000 	add	x0, x0, #0x30
     ac0:	bd4063e0 	ldr	s0, [sp, #96]
     ac4:	bd000000 	str	s0, [x0]
     ac8:	f9403fe0 	ldr	x0, [sp, #120]
     acc:	9100d000 	add	x0, x0, #0x34
     ad0:	bd4067e0 	ldr	s0, [sp, #100]
     ad4:	bd000000 	str	s0, [x0]
     ad8:	f9403fe0 	ldr	x0, [sp, #120]
     adc:	9100e000 	add	x0, x0, #0x38
     ae0:	bd406be0 	ldr	s0, [sp, #104]
     ae4:	bd000000 	str	s0, [x0]
     ae8:	f9403fe0 	ldr	x0, [sp, #120]
     aec:	9100f000 	add	x0, x0, #0x3c
     af0:	bd406fe0 	ldr	s0, [sp, #108]
     af4:	bd000000 	str	s0, [x0]
     af8:	f94043e0 	ldr	x0, [sp, #128]
     afc:	91004000 	add	x0, x0, #0x10
     b00:	f90043e0 	str	x0, [sp, #128]
     b04:	f94047e0 	ldr	x0, [sp, #136]
     b08:	91004000 	add	x0, x0, #0x10
     b0c:	f90047e0 	str	x0, [sp, #136]
     b10:	f9404be0 	ldr	x0, [sp, #144]
     b14:	91004000 	add	x0, x0, #0x10
     b18:	f9004be0 	str	x0, [sp, #144]
     b1c:	f9404fe0 	ldr	x0, [sp, #152]
     b20:	91004000 	add	x0, x0, #0x10
     b24:	f9004fe0 	str	x0, [sp, #152]
     b28:	b9401fe0 	ldr	w0, [sp, #28]
     b2c:	531e7400 	lsl	w0, w0, #2
     b30:	93407c00 	sxtw	x0, w0
     b34:	d37ef400 	lsl	x0, x0, #2
     b38:	f9403fe1 	ldr	x1, [sp, #120]
     b3c:	8b000020 	add	x0, x1, x0
     b40:	f9003fe0 	str	x0, [sp, #120]
     b44:	b9402fe0 	ldr	w0, [sp, #44]
     b48:	11001000 	add	w0, w0, #0x4
     b4c:	b9002fe0 	str	w0, [sp, #44]
     b50:	17ffff76 	b	928 <_Z5packBiiPfiS_+0xb4>
     b54:	b9400fe0 	ldr	w0, [sp, #12]
     b58:	531e7400 	lsl	w0, w0, #2
     b5c:	93407c00 	sxtw	x0, w0
     b60:	d37ef400 	lsl	x0, x0, #2
     b64:	f9403be1 	ldr	x1, [sp, #112]
     b68:	8b000020 	add	x0, x1, x0
     b6c:	f9003be0 	str	x0, [sp, #112]
     b70:	b9402be0 	ldr	w0, [sp, #40]
     b74:	11001000 	add	w0, w0, #0x4
     b78:	b9002be0 	str	w0, [sp, #40]
     b7c:	17ffff47 	b	898 <_Z5packBiiPfiS_+0x24>
     b80:	d503201f 	nop
     b84:	910283ff 	add	sp, sp, #0xa0
     b88:	d65f03c0 	ret

0000000000000b8c <_Z10kernel_4x4iiiPfS_S_i>:
     b8c:	d11343ff 	sub	sp, sp, #0x4d0
     b90:	a9007bfd 	stp	x29, x30, [sp]
     b94:	910003fd 	mov	x29, sp
     b98:	b9003fe0 	str	w0, [sp, #60]
     b9c:	b9003be1 	str	w1, [sp, #56]
     ba0:	b90037e2 	str	w2, [sp, #52]
     ba4:	f90017e3 	str	x3, [sp, #40]
     ba8:	f90013e4 	str	x4, [sp, #32]
     bac:	f9000fe5 	str	x5, [sp, #24]
     bb0:	b90033e6 	str	w6, [sp, #48]
     bb4:	90000000 	adrp	x0, 0 <__stack_chk_guard>
     bb8:	f9400000 	ldr	x0, [x0]
     bbc:	f9400001 	ldr	x1, [x0]
     bc0:	f90267e1 	str	x1, [sp, #1224]
     bc4:	d2800001 	mov	x1, #0x0                   	// #0
     bc8:	f94017e0 	ldr	x0, [sp, #40]
     bcc:	f9002fe0 	str	x0, [sp, #88]
     bd0:	f94013e0 	ldr	x0, [sp, #32]
     bd4:	f90033e0 	str	x0, [sp, #96]
     bd8:	f9400fe0 	ldr	x0, [sp, #24]
     bdc:	f90037e0 	str	x0, [sp, #104]
     be0:	b9004fff 	str	wzr, [sp, #76]
     be4:	b9404fe0 	ldr	w0, [sp, #76]
     be8:	11000c00 	add	w0, w0, #0x3
     bec:	b9403fe1 	ldr	w1, [sp, #60]
     bf0:	6b00003f 	cmp	w1, w0
     bf4:	54003b4d 	b.le	135c <_Z10kernel_4x4iiiPfS_S_i+0x7d0>
     bf8:	b90053ff 	str	wzr, [sp, #80]
     bfc:	b94053e0 	ldr	w0, [sp, #80]
     c00:	11000c00 	add	w0, w0, #0x3
     c04:	b9403be1 	ldr	w1, [sp, #56]
     c08:	6b00003f 	cmp	w1, w0
     c0c:	540037cd 	b.le	1304 <_Z10kernel_4x4iiiPfS_S_i+0x778>
     c10:	4f000400 	movi	v0.4s, #0x0
     c14:	3d8043e0 	str	q0, [sp, #256]
     c18:	4f000400 	movi	v0.4s, #0x0
     c1c:	3d8047e0 	str	q0, [sp, #272]
     c20:	4f000400 	movi	v0.4s, #0x0
     c24:	3d804be0 	str	q0, [sp, #288]
     c28:	4f000400 	movi	v0.4s, #0x0
     c2c:	3d804fe0 	str	q0, [sp, #304]
     c30:	f9402fe0 	ldr	x0, [sp, #88]
     c34:	f9800000 	prfm	pldl1keep, [x0]
     c38:	f94033e0 	ldr	x0, [sp, #96]
     c3c:	f9800000 	prfm	pldl1keep, [x0]
     c40:	b90057ff 	str	wzr, [sp, #84]
     c44:	b94057e0 	ldr	w0, [sp, #84]
     c48:	11000c00 	add	w0, w0, #0x3
     c4c:	b94037e1 	ldr	w1, [sp, #52]
     c50:	6b00003f 	cmp	w1, w0
     c54:	540024ed 	b.le	10f0 <_Z10kernel_4x4iiiPfS_S_i+0x564>
     c58:	f9402fe0 	ldr	x0, [sp, #88]
     c5c:	f90057e0 	str	x0, [sp, #168]
     c60:	f94057e0 	ldr	x0, [sp, #168]
     c64:	3dc00000 	ldr	q0, [x0]
     c68:	d503201f 	nop
     c6c:	3d8053e0 	str	q0, [sp, #320]
     c70:	f94033e0 	ldr	x0, [sp, #96]
     c74:	f90053e0 	str	x0, [sp, #160]
     c78:	f94053e0 	ldr	x0, [sp, #160]
     c7c:	3dc00000 	ldr	q0, [x0]
     c80:	d503201f 	nop
     c84:	3d8057e0 	str	q0, [sp, #336]
     c88:	3dc043e0 	ldr	q0, [sp, #256]
     c8c:	3d80fbe0 	str	q0, [sp, #992]
     c90:	3dc057e0 	ldr	q0, [sp, #336]
     c94:	3d80ffe0 	str	q0, [sp, #1008]
     c98:	3dc053e0 	ldr	q0, [sp, #320]
     c9c:	3d803fe0 	str	q0, [sp, #240]
     ca0:	bd40f3e0 	ldr	s0, [sp, #240]
     ca4:	4e040400 	dup	v0.4s, v0.s[0]
     ca8:	4ea01c01 	mov	v1.16b, v0.16b
     cac:	3dc0ffe0 	ldr	q0, [sp, #1008]
     cb0:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     cb4:	3dc0fbe0 	ldr	q0, [sp, #992]
     cb8:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     cbc:	3d8043e0 	str	q0, [sp, #256]
     cc0:	3dc047e0 	ldr	q0, [sp, #272]
     cc4:	3d80f3e0 	str	q0, [sp, #960]
     cc8:	3dc057e0 	ldr	q0, [sp, #336]
     ccc:	3d80f7e0 	str	q0, [sp, #976]
     cd0:	3dc053e0 	ldr	q0, [sp, #320]
     cd4:	3d803fe0 	str	q0, [sp, #240]
     cd8:	bd40f7e0 	ldr	s0, [sp, #244]
     cdc:	4e040400 	dup	v0.4s, v0.s[0]
     ce0:	4ea01c01 	mov	v1.16b, v0.16b
     ce4:	3dc0f7e0 	ldr	q0, [sp, #976]
     ce8:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     cec:	3dc0f3e0 	ldr	q0, [sp, #960]
     cf0:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     cf4:	3d8047e0 	str	q0, [sp, #272]
     cf8:	3dc04be0 	ldr	q0, [sp, #288]
     cfc:	3d80ebe0 	str	q0, [sp, #928]
     d00:	3dc057e0 	ldr	q0, [sp, #336]
     d04:	3d80efe0 	str	q0, [sp, #944]
     d08:	3dc053e0 	ldr	q0, [sp, #320]
     d0c:	3d803fe0 	str	q0, [sp, #240]
     d10:	bd40fbe0 	ldr	s0, [sp, #248]
     d14:	4e040400 	dup	v0.4s, v0.s[0]
     d18:	4ea01c01 	mov	v1.16b, v0.16b
     d1c:	3dc0efe0 	ldr	q0, [sp, #944]
     d20:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     d24:	3dc0ebe0 	ldr	q0, [sp, #928]
     d28:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     d2c:	3d804be0 	str	q0, [sp, #288]
     d30:	3dc04fe0 	ldr	q0, [sp, #304]
     d34:	3d80e3e0 	str	q0, [sp, #896]
     d38:	3dc057e0 	ldr	q0, [sp, #336]
     d3c:	3d80e7e0 	str	q0, [sp, #912]
     d40:	3dc053e0 	ldr	q0, [sp, #320]
     d44:	3d803fe0 	str	q0, [sp, #240]
     d48:	bd40ffe0 	ldr	s0, [sp, #252]
     d4c:	4e040400 	dup	v0.4s, v0.s[0]
     d50:	4ea01c01 	mov	v1.16b, v0.16b
     d54:	3dc0e7e0 	ldr	q0, [sp, #912]
     d58:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     d5c:	3dc0e3e0 	ldr	q0, [sp, #896]
     d60:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     d64:	3d804fe0 	str	q0, [sp, #304]
     d68:	f9402fe0 	ldr	x0, [sp, #88]
     d6c:	91004000 	add	x0, x0, #0x10
     d70:	f9004fe0 	str	x0, [sp, #152]
     d74:	f9404fe0 	ldr	x0, [sp, #152]
     d78:	3dc00000 	ldr	q0, [x0]
     d7c:	d503201f 	nop
     d80:	3d805be0 	str	q0, [sp, #352]
     d84:	f94033e0 	ldr	x0, [sp, #96]
     d88:	91004000 	add	x0, x0, #0x10
     d8c:	f9004be0 	str	x0, [sp, #144]
     d90:	f9404be0 	ldr	x0, [sp, #144]
     d94:	3dc00000 	ldr	q0, [x0]
     d98:	d503201f 	nop
     d9c:	3d805fe0 	str	q0, [sp, #368]
     da0:	3dc043e0 	ldr	q0, [sp, #256]
     da4:	3d80dbe0 	str	q0, [sp, #864]
     da8:	3dc05fe0 	ldr	q0, [sp, #368]
     dac:	3d80dfe0 	str	q0, [sp, #880]
     db0:	3dc05be0 	ldr	q0, [sp, #352]
     db4:	3d803fe0 	str	q0, [sp, #240]
     db8:	bd40f3e0 	ldr	s0, [sp, #240]
     dbc:	4e040400 	dup	v0.4s, v0.s[0]
     dc0:	4ea01c01 	mov	v1.16b, v0.16b
     dc4:	3dc0dfe0 	ldr	q0, [sp, #880]
     dc8:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     dcc:	3dc0dbe0 	ldr	q0, [sp, #864]
     dd0:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     dd4:	3d8043e0 	str	q0, [sp, #256]
     dd8:	3dc047e0 	ldr	q0, [sp, #272]
     ddc:	3d80d3e0 	str	q0, [sp, #832]
     de0:	3dc05fe0 	ldr	q0, [sp, #368]
     de4:	3d80d7e0 	str	q0, [sp, #848]
     de8:	3dc05be0 	ldr	q0, [sp, #352]
     dec:	3d803fe0 	str	q0, [sp, #240]
     df0:	bd40f7e0 	ldr	s0, [sp, #244]
     df4:	4e040400 	dup	v0.4s, v0.s[0]
     df8:	4ea01c01 	mov	v1.16b, v0.16b
     dfc:	3dc0d7e0 	ldr	q0, [sp, #848]
     e00:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     e04:	3dc0d3e0 	ldr	q0, [sp, #832]
     e08:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     e0c:	3d8047e0 	str	q0, [sp, #272]
     e10:	3dc04be0 	ldr	q0, [sp, #288]
     e14:	3d80cbe0 	str	q0, [sp, #800]
     e18:	3dc05fe0 	ldr	q0, [sp, #368]
     e1c:	3d80cfe0 	str	q0, [sp, #816]
     e20:	3dc05be0 	ldr	q0, [sp, #352]
     e24:	3d803fe0 	str	q0, [sp, #240]
     e28:	bd40fbe0 	ldr	s0, [sp, #248]
     e2c:	4e040400 	dup	v0.4s, v0.s[0]
     e30:	4ea01c01 	mov	v1.16b, v0.16b
     e34:	3dc0cfe0 	ldr	q0, [sp, #816]
     e38:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     e3c:	3dc0cbe0 	ldr	q0, [sp, #800]
     e40:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     e44:	3d804be0 	str	q0, [sp, #288]
     e48:	3dc04fe0 	ldr	q0, [sp, #304]
     e4c:	3d80c3e0 	str	q0, [sp, #768]
     e50:	3dc05fe0 	ldr	q0, [sp, #368]
     e54:	3d80c7e0 	str	q0, [sp, #784]
     e58:	3dc05be0 	ldr	q0, [sp, #352]
     e5c:	3d803fe0 	str	q0, [sp, #240]
     e60:	bd40ffe0 	ldr	s0, [sp, #252]
     e64:	4e040400 	dup	v0.4s, v0.s[0]
     e68:	4ea01c01 	mov	v1.16b, v0.16b
     e6c:	3dc0c7e0 	ldr	q0, [sp, #784]
     e70:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     e74:	3dc0c3e0 	ldr	q0, [sp, #768]
     e78:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     e7c:	3d804fe0 	str	q0, [sp, #304]
     e80:	f9402fe0 	ldr	x0, [sp, #88]
     e84:	91008000 	add	x0, x0, #0x20
     e88:	f90047e0 	str	x0, [sp, #136]
     e8c:	f94047e0 	ldr	x0, [sp, #136]
     e90:	3dc00000 	ldr	q0, [x0]
     e94:	d503201f 	nop
     e98:	3d8063e0 	str	q0, [sp, #384]
     e9c:	f94033e0 	ldr	x0, [sp, #96]
     ea0:	91008000 	add	x0, x0, #0x20
     ea4:	f90043e0 	str	x0, [sp, #128]
     ea8:	f94043e0 	ldr	x0, [sp, #128]
     eac:	3dc00000 	ldr	q0, [x0]
     eb0:	d503201f 	nop
     eb4:	3d8067e0 	str	q0, [sp, #400]
     eb8:	3dc043e0 	ldr	q0, [sp, #256]
     ebc:	3d80bbe0 	str	q0, [sp, #736]
     ec0:	3dc067e0 	ldr	q0, [sp, #400]
     ec4:	3d80bfe0 	str	q0, [sp, #752]
     ec8:	3dc063e0 	ldr	q0, [sp, #384]
     ecc:	3d803fe0 	str	q0, [sp, #240]
     ed0:	bd40f3e0 	ldr	s0, [sp, #240]
     ed4:	4e040400 	dup	v0.4s, v0.s[0]
     ed8:	4ea01c01 	mov	v1.16b, v0.16b
     edc:	3dc0bfe0 	ldr	q0, [sp, #752]
     ee0:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     ee4:	3dc0bbe0 	ldr	q0, [sp, #736]
     ee8:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     eec:	3d8043e0 	str	q0, [sp, #256]
     ef0:	3dc047e0 	ldr	q0, [sp, #272]
     ef4:	3d80b3e0 	str	q0, [sp, #704]
     ef8:	3dc067e0 	ldr	q0, [sp, #400]
     efc:	3d80b7e0 	str	q0, [sp, #720]
     f00:	3dc063e0 	ldr	q0, [sp, #384]
     f04:	3d803fe0 	str	q0, [sp, #240]
     f08:	bd40f7e0 	ldr	s0, [sp, #244]
     f0c:	4e040400 	dup	v0.4s, v0.s[0]
     f10:	4ea01c01 	mov	v1.16b, v0.16b
     f14:	3dc0b7e0 	ldr	q0, [sp, #720]
     f18:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     f1c:	3dc0b3e0 	ldr	q0, [sp, #704]
     f20:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     f24:	3d8047e0 	str	q0, [sp, #272]
     f28:	3dc04be0 	ldr	q0, [sp, #288]
     f2c:	3d80abe0 	str	q0, [sp, #672]
     f30:	3dc067e0 	ldr	q0, [sp, #400]
     f34:	3d80afe0 	str	q0, [sp, #688]
     f38:	3dc063e0 	ldr	q0, [sp, #384]
     f3c:	3d803fe0 	str	q0, [sp, #240]
     f40:	bd40fbe0 	ldr	s0, [sp, #248]
     f44:	4e040400 	dup	v0.4s, v0.s[0]
     f48:	4ea01c01 	mov	v1.16b, v0.16b
     f4c:	3dc0afe0 	ldr	q0, [sp, #688]
     f50:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     f54:	3dc0abe0 	ldr	q0, [sp, #672]
     f58:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     f5c:	3d804be0 	str	q0, [sp, #288]
     f60:	3dc04fe0 	ldr	q0, [sp, #304]
     f64:	3d80a3e0 	str	q0, [sp, #640]
     f68:	3dc067e0 	ldr	q0, [sp, #400]
     f6c:	3d80a7e0 	str	q0, [sp, #656]
     f70:	3dc063e0 	ldr	q0, [sp, #384]
     f74:	3d803fe0 	str	q0, [sp, #240]
     f78:	bd40ffe0 	ldr	s0, [sp, #252]
     f7c:	4e040400 	dup	v0.4s, v0.s[0]
     f80:	4ea01c01 	mov	v1.16b, v0.16b
     f84:	3dc0a7e0 	ldr	q0, [sp, #656]
     f88:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     f8c:	3dc0a3e0 	ldr	q0, [sp, #640]
     f90:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     f94:	3d804fe0 	str	q0, [sp, #304]
     f98:	f9402fe0 	ldr	x0, [sp, #88]
     f9c:	9100c000 	add	x0, x0, #0x30
     fa0:	f9003fe0 	str	x0, [sp, #120]
     fa4:	f9403fe0 	ldr	x0, [sp, #120]
     fa8:	3dc00000 	ldr	q0, [x0]
     fac:	d503201f 	nop
     fb0:	3d806be0 	str	q0, [sp, #416]
     fb4:	f94033e0 	ldr	x0, [sp, #96]
     fb8:	9100c000 	add	x0, x0, #0x30
     fbc:	f9003be0 	str	x0, [sp, #112]
     fc0:	f9403be0 	ldr	x0, [sp, #112]
     fc4:	3dc00000 	ldr	q0, [x0]
     fc8:	d503201f 	nop
     fcc:	3d806fe0 	str	q0, [sp, #432]
     fd0:	3dc043e0 	ldr	q0, [sp, #256]
     fd4:	3d809be0 	str	q0, [sp, #608]
     fd8:	3dc06fe0 	ldr	q0, [sp, #432]
     fdc:	3d809fe0 	str	q0, [sp, #624]
     fe0:	3dc06be0 	ldr	q0, [sp, #416]
     fe4:	3d803fe0 	str	q0, [sp, #240]
     fe8:	bd40f3e0 	ldr	s0, [sp, #240]
     fec:	4e040400 	dup	v0.4s, v0.s[0]
     ff0:	4ea01c01 	mov	v1.16b, v0.16b
     ff4:	3dc09fe0 	ldr	q0, [sp, #624]
     ff8:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
     ffc:	3dc09be0 	ldr	q0, [sp, #608]
    1000:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    1004:	3d8043e0 	str	q0, [sp, #256]
    1008:	3dc047e0 	ldr	q0, [sp, #272]
    100c:	3d8093e0 	str	q0, [sp, #576]
    1010:	3dc06fe0 	ldr	q0, [sp, #432]
    1014:	3d8097e0 	str	q0, [sp, #592]
    1018:	3dc06be0 	ldr	q0, [sp, #416]
    101c:	3d803fe0 	str	q0, [sp, #240]
    1020:	bd40f7e0 	ldr	s0, [sp, #244]
    1024:	4e040400 	dup	v0.4s, v0.s[0]
    1028:	4ea01c01 	mov	v1.16b, v0.16b
    102c:	3dc097e0 	ldr	q0, [sp, #592]
    1030:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
    1034:	3dc093e0 	ldr	q0, [sp, #576]
    1038:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    103c:	3d8047e0 	str	q0, [sp, #272]
    1040:	3dc04be0 	ldr	q0, [sp, #288]
    1044:	3d808be0 	str	q0, [sp, #544]
    1048:	3dc06fe0 	ldr	q0, [sp, #432]
    104c:	3d808fe0 	str	q0, [sp, #560]
    1050:	3dc06be0 	ldr	q0, [sp, #416]
    1054:	3d803fe0 	str	q0, [sp, #240]
    1058:	bd40fbe0 	ldr	s0, [sp, #248]
    105c:	4e040400 	dup	v0.4s, v0.s[0]
    1060:	4ea01c01 	mov	v1.16b, v0.16b
    1064:	3dc08fe0 	ldr	q0, [sp, #560]
    1068:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
    106c:	3dc08be0 	ldr	q0, [sp, #544]
    1070:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    1074:	3d804be0 	str	q0, [sp, #288]
    1078:	3dc04fe0 	ldr	q0, [sp, #304]
    107c:	3d8083e0 	str	q0, [sp, #512]
    1080:	3dc06fe0 	ldr	q0, [sp, #432]
    1084:	3d8087e0 	str	q0, [sp, #528]
    1088:	3dc06be0 	ldr	q0, [sp, #416]
    108c:	3d803fe0 	str	q0, [sp, #240]
    1090:	bd40ffe0 	ldr	s0, [sp, #252]
    1094:	4e040400 	dup	v0.4s, v0.s[0]
    1098:	4ea01c01 	mov	v1.16b, v0.16b
    109c:	3dc087e0 	ldr	q0, [sp, #528]
    10a0:	6e20dc21 	fmul	v1.4s, v1.4s, v0.4s
    10a4:	3dc083e0 	ldr	q0, [sp, #512]
    10a8:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    10ac:	3d804fe0 	str	q0, [sp, #304]
    10b0:	f9402fe0 	ldr	x0, [sp, #88]
    10b4:	91010000 	add	x0, x0, #0x40
    10b8:	f9800000 	prfm	pldl1keep, [x0]
    10bc:	f94033e0 	ldr	x0, [sp, #96]
    10c0:	91010000 	add	x0, x0, #0x40
    10c4:	f9800000 	prfm	pldl1keep, [x0]
    10c8:	f9402fe0 	ldr	x0, [sp, #88]
    10cc:	91010000 	add	x0, x0, #0x40
    10d0:	f9002fe0 	str	x0, [sp, #88]
    10d4:	f94033e0 	ldr	x0, [sp, #96]
    10d8:	91010000 	add	x0, x0, #0x40
    10dc:	f90033e0 	str	x0, [sp, #96]
    10e0:	b94057e0 	ldr	w0, [sp, #84]
    10e4:	11001000 	add	w0, w0, #0x4
    10e8:	b90057e0 	str	w0, [sp, #84]
    10ec:	17fffed6 	b	c44 <_Z10kernel_4x4iiiPfS_S_i+0xb8>
    10f0:	b94037e0 	ldr	w0, [sp, #52]
    10f4:	531e7400 	lsl	w0, w0, #2
    10f8:	93407c00 	sxtw	x0, w0
    10fc:	d37ef400 	lsl	x0, x0, #2
    1100:	cb0003e0 	neg	x0, x0
    1104:	f9402fe1 	ldr	x1, [sp, #88]
    1108:	8b000020 	add	x0, x1, x0
    110c:	f9002fe0 	str	x0, [sp, #88]
    1110:	f94037e0 	ldr	x0, [sp, #104]
    1114:	f90077e0 	str	x0, [sp, #232]
    1118:	f94077e0 	ldr	x0, [sp, #232]
    111c:	3dc00000 	ldr	q0, [x0]
    1120:	d503201f 	nop
    1124:	3d8073e0 	str	q0, [sp, #448]
    1128:	b98033e0 	ldrsw	x0, [sp, #48]
    112c:	d37ef400 	lsl	x0, x0, #2
    1130:	f94037e1 	ldr	x1, [sp, #104]
    1134:	8b000020 	add	x0, x1, x0
    1138:	f90073e0 	str	x0, [sp, #224]
    113c:	f94073e0 	ldr	x0, [sp, #224]
    1140:	3dc00000 	ldr	q0, [x0]
    1144:	d503201f 	nop
    1148:	3d8077e0 	str	q0, [sp, #464]
    114c:	b94033e0 	ldr	w0, [sp, #48]
    1150:	531f7800 	lsl	w0, w0, #1
    1154:	93407c00 	sxtw	x0, w0
    1158:	d37ef400 	lsl	x0, x0, #2
    115c:	f94037e1 	ldr	x1, [sp, #104]
    1160:	8b000020 	add	x0, x1, x0
    1164:	f9006fe0 	str	x0, [sp, #216]
    1168:	f9406fe0 	ldr	x0, [sp, #216]
    116c:	3dc00000 	ldr	q0, [x0]
    1170:	d503201f 	nop
    1174:	3d807be0 	str	q0, [sp, #480]
    1178:	b94033e1 	ldr	w1, [sp, #48]
    117c:	2a0103e0 	mov	w0, w1
    1180:	531f7800 	lsl	w0, w0, #1
    1184:	0b010000 	add	w0, w0, w1
    1188:	93407c00 	sxtw	x0, w0
    118c:	d37ef400 	lsl	x0, x0, #2
    1190:	f94037e1 	ldr	x1, [sp, #104]
    1194:	8b000020 	add	x0, x1, x0
    1198:	f9006be0 	str	x0, [sp, #208]
    119c:	f9406be0 	ldr	x0, [sp, #208]
    11a0:	3dc00000 	ldr	q0, [x0]
    11a4:	d503201f 	nop
    11a8:	3d807fe0 	str	q0, [sp, #496]
    11ac:	3dc073e0 	ldr	q0, [sp, #448]
    11b0:	3d812be0 	str	q0, [sp, #1184]
    11b4:	3dc043e0 	ldr	q0, [sp, #256]
    11b8:	3d812fe0 	str	q0, [sp, #1200]
    11bc:	3dc12be1 	ldr	q1, [sp, #1184]
    11c0:	3dc12fe0 	ldr	q0, [sp, #1200]
    11c4:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    11c8:	3d8073e0 	str	q0, [sp, #448]
    11cc:	3dc077e0 	ldr	q0, [sp, #464]
    11d0:	3d8123e0 	str	q0, [sp, #1152]
    11d4:	3dc047e0 	ldr	q0, [sp, #272]
    11d8:	3d8127e0 	str	q0, [sp, #1168]
    11dc:	3dc123e1 	ldr	q1, [sp, #1152]
    11e0:	3dc127e0 	ldr	q0, [sp, #1168]
    11e4:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    11e8:	3d8077e0 	str	q0, [sp, #464]
    11ec:	3dc07be0 	ldr	q0, [sp, #480]
    11f0:	3d811be0 	str	q0, [sp, #1120]
    11f4:	3dc04be0 	ldr	q0, [sp, #288]
    11f8:	3d811fe0 	str	q0, [sp, #1136]
    11fc:	3dc11be1 	ldr	q1, [sp, #1120]
    1200:	3dc11fe0 	ldr	q0, [sp, #1136]
    1204:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    1208:	3d807be0 	str	q0, [sp, #480]
    120c:	3dc07fe0 	ldr	q0, [sp, #496]
    1210:	3d8113e0 	str	q0, [sp, #1088]
    1214:	3dc04fe0 	ldr	q0, [sp, #304]
    1218:	3d8117e0 	str	q0, [sp, #1104]
    121c:	3dc113e1 	ldr	q1, [sp, #1088]
    1220:	3dc117e0 	ldr	q0, [sp, #1104]
    1224:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
    1228:	3d807fe0 	str	q0, [sp, #496]
    122c:	f94037e0 	ldr	x0, [sp, #104]
    1230:	f90067e0 	str	x0, [sp, #200]
    1234:	3dc073e0 	ldr	q0, [sp, #448]
    1238:	3d810fe0 	str	q0, [sp, #1072]
    123c:	f94067e0 	ldr	x0, [sp, #200]
    1240:	3dc10fe0 	ldr	q0, [sp, #1072]
    1244:	3d800000 	str	q0, [x0]
    1248:	d503201f 	nop
    124c:	b98033e0 	ldrsw	x0, [sp, #48]
    1250:	d37ef400 	lsl	x0, x0, #2
    1254:	f94037e1 	ldr	x1, [sp, #104]
    1258:	8b000020 	add	x0, x1, x0
    125c:	f90063e0 	str	x0, [sp, #192]
    1260:	3dc077e0 	ldr	q0, [sp, #464]
    1264:	3d810be0 	str	q0, [sp, #1056]
    1268:	f94063e0 	ldr	x0, [sp, #192]
    126c:	3dc10be0 	ldr	q0, [sp, #1056]
    1270:	3d800000 	str	q0, [x0]
    1274:	d503201f 	nop
    1278:	b94033e0 	ldr	w0, [sp, #48]
    127c:	531f7800 	lsl	w0, w0, #1
    1280:	93407c00 	sxtw	x0, w0
    1284:	d37ef400 	lsl	x0, x0, #2
    1288:	f94037e1 	ldr	x1, [sp, #104]
    128c:	8b000020 	add	x0, x1, x0
    1290:	f9005fe0 	str	x0, [sp, #184]
    1294:	3dc07be0 	ldr	q0, [sp, #480]
    1298:	3d8107e0 	str	q0, [sp, #1040]
    129c:	f9405fe0 	ldr	x0, [sp, #184]
    12a0:	3dc107e0 	ldr	q0, [sp, #1040]
    12a4:	3d800000 	str	q0, [x0]
    12a8:	d503201f 	nop
    12ac:	b94033e1 	ldr	w1, [sp, #48]
    12b0:	2a0103e0 	mov	w0, w1
    12b4:	531f7800 	lsl	w0, w0, #1
    12b8:	0b010000 	add	w0, w0, w1
    12bc:	93407c00 	sxtw	x0, w0
    12c0:	d37ef400 	lsl	x0, x0, #2
    12c4:	f94037e1 	ldr	x1, [sp, #104]
    12c8:	8b000020 	add	x0, x1, x0
    12cc:	f9005be0 	str	x0, [sp, #176]
    12d0:	3dc07fe0 	ldr	q0, [sp, #496]
    12d4:	3d8103e0 	str	q0, [sp, #1024]
    12d8:	f9405be0 	ldr	x0, [sp, #176]
    12dc:	3dc103e0 	ldr	q0, [sp, #1024]
    12e0:	3d800000 	str	q0, [x0]
    12e4:	d503201f 	nop
    12e8:	f94037e0 	ldr	x0, [sp, #104]
    12ec:	91004000 	add	x0, x0, #0x10
    12f0:	f90037e0 	str	x0, [sp, #104]
    12f4:	b94053e0 	ldr	w0, [sp, #80]
    12f8:	11001000 	add	w0, w0, #0x4
    12fc:	b90053e0 	str	w0, [sp, #80]
    1300:	17fffe3f 	b	bfc <_Z10kernel_4x4iiiPfS_S_i+0x70>
    1304:	f94013e0 	ldr	x0, [sp, #32]
    1308:	f90033e0 	str	x0, [sp, #96]
    130c:	b94037e0 	ldr	w0, [sp, #52]
    1310:	531e7400 	lsl	w0, w0, #2
    1314:	93407c00 	sxtw	x0, w0
    1318:	d37ef400 	lsl	x0, x0, #2
    131c:	f9402fe1 	ldr	x1, [sp, #88]
    1320:	8b000020 	add	x0, x1, x0
    1324:	f9002fe0 	str	x0, [sp, #88]
    1328:	b94033e0 	ldr	w0, [sp, #48]
    132c:	531e7400 	lsl	w0, w0, #2
    1330:	93407c00 	sxtw	x0, w0
    1334:	d37ef400 	lsl	x0, x0, #2
    1338:	f9400fe1 	ldr	x1, [sp, #24]
    133c:	8b000020 	add	x0, x1, x0
    1340:	f9000fe0 	str	x0, [sp, #24]
    1344:	f9400fe0 	ldr	x0, [sp, #24]
    1348:	f90037e0 	str	x0, [sp, #104]
    134c:	b9404fe0 	ldr	w0, [sp, #76]
    1350:	11001000 	add	w0, w0, #0x4
    1354:	b9004fe0 	str	w0, [sp, #76]
    1358:	17fffe23 	b	be4 <_Z10kernel_4x4iiiPfS_S_i+0x58>
    135c:	d503201f 	nop
    1360:	90000000 	adrp	x0, 0 <__stack_chk_guard>
    1364:	f9400000 	ldr	x0, [x0]
    1368:	f94267e1 	ldr	x1, [sp, #1224]
    136c:	f9400002 	ldr	x2, [x0]
    1370:	eb020021 	subs	x1, x1, x2
    1374:	d2800002 	mov	x2, #0x0                   	// #0
    1378:	54000040 	b.eq	1380 <_Z10kernel_4x4iiiPfS_S_i+0x7f4>  // b.none
    137c:	94000000 	bl	0 <__stack_chk_fail>
    1380:	a9407bfd 	ldp	x29, x30, [sp]
    1384:	911343ff 	add	sp, sp, #0x4d0
    1388:	d65f03c0 	ret
