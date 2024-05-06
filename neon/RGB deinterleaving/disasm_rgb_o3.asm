
exe_rgb_o3:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000000658 <_init>:
 658:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
 65c:	910003fd 	mov	x29, sp
 660:	9400002e 	bl	718 <call_weak_fn>
 664:	a8c17bfd 	ldp	x29, x30, [sp], #16
 668:	d65f03c0 	ret

Disassembly of section .plt:

0000000000000670 <.plt>:
 670:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
 674:	90000090 	adrp	x16, 10000 <__FRAME_END__+0xf478>
 678:	f947c611 	ldr	x17, [x16, #3976]
 67c:	913e2210 	add	x16, x16, #0xf88
 680:	d61f0220 	br	x17
 684:	d503201f 	nop
 688:	d503201f 	nop
 68c:	d503201f 	nop

0000000000000690 <__cxa_finalize@plt>:
 690:	90000090 	adrp	x16, 10000 <__FRAME_END__+0xf478>
 694:	f947ca11 	ldr	x17, [x16, #3984]
 698:	913e4210 	add	x16, x16, #0xf90
 69c:	d61f0220 	br	x17

00000000000006a0 <__libc_start_main@plt>:
 6a0:	90000090 	adrp	x16, 10000 <__FRAME_END__+0xf478>
 6a4:	f947ce11 	ldr	x17, [x16, #3992]
 6a8:	913e6210 	add	x16, x16, #0xf98
 6ac:	d61f0220 	br	x17

00000000000006b0 <__stack_chk_fail@plt>:
 6b0:	90000090 	adrp	x16, 10000 <__FRAME_END__+0xf478>
 6b4:	f947d211 	ldr	x17, [x16, #4000]
 6b8:	913e8210 	add	x16, x16, #0xfa0
 6bc:	d61f0220 	br	x17

00000000000006c0 <__gmon_start__@plt>:
 6c0:	90000090 	adrp	x16, 10000 <__FRAME_END__+0xf478>
 6c4:	f947d611 	ldr	x17, [x16, #4008]
 6c8:	913ea210 	add	x16, x16, #0xfa8
 6cc:	d61f0220 	br	x17

00000000000006d0 <abort@plt>:
 6d0:	90000090 	adrp	x16, 10000 <__FRAME_END__+0xf478>
 6d4:	f947da11 	ldr	x17, [x16, #4016]
 6d8:	913ec210 	add	x16, x16, #0xfb0
 6dc:	d61f0220 	br	x17

Disassembly of section .text:

00000000000006e0 <_start>:
 6e0:	d280001d 	mov	x29, #0x0                   	// #0
 6e4:	d280001e 	mov	x30, #0x0                   	// #0
 6e8:	aa0003e5 	mov	x5, x0
 6ec:	f94003e1 	ldr	x1, [sp]
 6f0:	910023e2 	add	x2, sp, #0x8
 6f4:	910003e6 	mov	x6, sp
 6f8:	90000080 	adrp	x0, 10000 <__FRAME_END__+0xf478>
 6fc:	f947f800 	ldr	x0, [x0, #4080]
 700:	90000083 	adrp	x3, 10000 <__FRAME_END__+0xf478>
 704:	f947f463 	ldr	x3, [x3, #4072]
 708:	90000084 	adrp	x4, 10000 <__FRAME_END__+0xf478>
 70c:	f947e084 	ldr	x4, [x4, #4032]
 710:	97ffffe4 	bl	6a0 <__libc_start_main@plt>
 714:	97ffffef 	bl	6d0 <abort@plt>

0000000000000718 <call_weak_fn>:
 718:	90000080 	adrp	x0, 10000 <__FRAME_END__+0xf478>
 71c:	f947ec00 	ldr	x0, [x0, #4056]
 720:	b4000040 	cbz	x0, 728 <call_weak_fn+0x10>
 724:	17ffffe7 	b	6c0 <__gmon_start__@plt>
 728:	d65f03c0 	ret
 72c:	d503201f 	nop

0000000000000730 <deregister_tm_clones>:
 730:	b0000080 	adrp	x0, 11000 <__data_start>
 734:	91004000 	add	x0, x0, #0x10
 738:	b0000081 	adrp	x1, 11000 <__data_start>
 73c:	91004021 	add	x1, x1, #0x10
 740:	eb00003f 	cmp	x1, x0
 744:	540000c0 	b.eq	75c <deregister_tm_clones+0x2c>  // b.none
 748:	90000081 	adrp	x1, 10000 <__FRAME_END__+0xf478>
 74c:	f947e421 	ldr	x1, [x1, #4040]
 750:	b4000061 	cbz	x1, 75c <deregister_tm_clones+0x2c>
 754:	aa0103f0 	mov	x16, x1
 758:	d61f0200 	br	x16
 75c:	d65f03c0 	ret

0000000000000760 <register_tm_clones>:
 760:	b0000080 	adrp	x0, 11000 <__data_start>
 764:	91004000 	add	x0, x0, #0x10
 768:	b0000081 	adrp	x1, 11000 <__data_start>
 76c:	91004021 	add	x1, x1, #0x10
 770:	cb000021 	sub	x1, x1, x0
 774:	d37ffc22 	lsr	x2, x1, #63
 778:	8b810c41 	add	x1, x2, x1, asr #3
 77c:	eb8107ff 	cmp	xzr, x1, asr #1
 780:	9341fc21 	asr	x1, x1, #1
 784:	540000c0 	b.eq	79c <register_tm_clones+0x3c>  // b.none
 788:	90000082 	adrp	x2, 10000 <__FRAME_END__+0xf478>
 78c:	f947fc42 	ldr	x2, [x2, #4088]
 790:	b4000062 	cbz	x2, 79c <register_tm_clones+0x3c>
 794:	aa0203f0 	mov	x16, x2
 798:	d61f0200 	br	x16
 79c:	d65f03c0 	ret

00000000000007a0 <__do_global_dtors_aux>:
 7a0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
 7a4:	910003fd 	mov	x29, sp
 7a8:	f9000bf3 	str	x19, [sp, #16]
 7ac:	b0000093 	adrp	x19, 11000 <__data_start>
 7b0:	39404260 	ldrb	w0, [x19, #16]
 7b4:	35000140 	cbnz	w0, 7dc <__do_global_dtors_aux+0x3c>
 7b8:	90000080 	adrp	x0, 10000 <__FRAME_END__+0xf478>
 7bc:	f947e800 	ldr	x0, [x0, #4048]
 7c0:	b4000080 	cbz	x0, 7d0 <__do_global_dtors_aux+0x30>
 7c4:	b0000080 	adrp	x0, 11000 <__data_start>
 7c8:	f9400400 	ldr	x0, [x0, #8]
 7cc:	97ffffb1 	bl	690 <__cxa_finalize@plt>
 7d0:	97ffffd8 	bl	730 <deregister_tm_clones>
 7d4:	52800020 	mov	w0, #0x1                   	// #1
 7d8:	39004260 	strb	w0, [x19, #16]
 7dc:	f9400bf3 	ldr	x19, [sp, #16]
 7e0:	a8c27bfd 	ldp	x29, x30, [sp], #32
 7e4:	d65f03c0 	ret

00000000000007e8 <frame_dummy>:
 7e8:	17ffffde 	b	760 <register_tm_clones>

00000000000007ec <rgb_deinterleave_neon>:
 7ec:	a9ac7bfd 	stp	x29, x30, [sp, #-320]!
 7f0:	910003fd 	mov	x29, sp
 7f4:	f9001fe0 	str	x0, [sp, #56]
 7f8:	f9001be1 	str	x1, [sp, #48]
 7fc:	f90017e2 	str	x2, [sp, #40]
 800:	f90013e3 	str	x3, [sp, #32]
 804:	b9001fe4 	str	w4, [sp, #28]
 808:	90000080 	adrp	x0, 10000 <__FRAME_END__+0xf478>
 80c:	f947f000 	ldr	x0, [x0, #4064]
 810:	f9400001 	ldr	x1, [x0]
 814:	f9009fe1 	str	x1, [sp, #312]
 818:	d2800001 	mov	x1, #0x0                   	// #0
 81c:	b9401fe0 	ldr	w0, [sp, #28]
 820:	11003c01 	add	w1, w0, #0xf
 824:	7100001f 	cmp	w0, #0x0
 828:	1a80b020 	csel	w0, w1, w0, lt  // lt = tstop
 82c:	13047c00 	asr	w0, w0, #4
 830:	b9004fe0 	str	w0, [sp, #76]
 834:	b9004bff 	str	wzr, [sp, #72]
 838:	1400004b 	b	964 <rgb_deinterleave_neon+0x178>
 83c:	b9404be1 	ldr	w1, [sp, #72]
 840:	2a0103e0 	mov	w0, w1
 844:	531f7800 	lsl	w0, w0, #1
 848:	0b010000 	add	w0, w0, w1
 84c:	531c6c00 	lsl	w0, w0, #4
 850:	93407c00 	sxtw	x0, w0
 854:	f94013e1 	ldr	x1, [sp, #32]
 858:	8b000020 	add	x0, x1, x0
 85c:	f90037e0 	str	x0, [sp, #104]
 860:	910283e0 	add	x0, sp, #0xa0
 864:	f94037e1 	ldr	x1, [sp, #104]
 868:	4c404021 	ld3	{v1.16b-v3.16b}, [x1]
 86c:	4c006001 	st1	{v1.16b-v3.16b}, [x0]
 870:	910283e0 	add	x0, sp, #0xa0
 874:	4c406001 	ld1	{v1.16b-v3.16b}, [x0]
 878:	4ea11c20 	mov	v0.16b, v1.16b
 87c:	910403e0 	add	x0, sp, #0x100
 880:	3d800000 	str	q0, [x0]
 884:	910283e0 	add	x0, sp, #0xa0
 888:	4c406001 	ld1	{v1.16b-v3.16b}, [x0]
 88c:	4ea21c40 	mov	v0.16b, v2.16b
 890:	910403e0 	add	x0, sp, #0x100
 894:	3d800400 	str	q0, [x0, #16]
 898:	910283e0 	add	x0, sp, #0xa0
 89c:	4c406001 	ld1	{v1.16b-v3.16b}, [x0]
 8a0:	4ea31c60 	mov	v0.16b, v3.16b
 8a4:	910403e0 	add	x0, sp, #0x100
 8a8:	3d800800 	str	q0, [x0, #32]
 8ac:	910403e0 	add	x0, sp, #0x100
 8b0:	4c406001 	ld1	{v1.16b-v3.16b}, [x0]
 8b4:	910343e0 	add	x0, sp, #0xd0
 8b8:	4c006001 	st1	{v1.16b-v3.16b}, [x0]
 8bc:	b9404be0 	ldr	w0, [sp, #72]
 8c0:	531c6c00 	lsl	w0, w0, #4
 8c4:	93407c00 	sxtw	x0, w0
 8c8:	f9401fe1 	ldr	x1, [sp, #56]
 8cc:	8b000020 	add	x0, x1, x0
 8d0:	910343e1 	add	x1, sp, #0xd0
 8d4:	3dc00020 	ldr	q0, [x1]
 8d8:	f90033e0 	str	x0, [sp, #96]
 8dc:	3d8027e0 	str	q0, [sp, #144]
 8e0:	3dc027e0 	ldr	q0, [sp, #144]
 8e4:	f94033e0 	ldr	x0, [sp, #96]
 8e8:	3d800000 	str	q0, [x0]
 8ec:	d503201f 	nop
 8f0:	b9404be0 	ldr	w0, [sp, #72]
 8f4:	531c6c00 	lsl	w0, w0, #4
 8f8:	93407c00 	sxtw	x0, w0
 8fc:	f9401be1 	ldr	x1, [sp, #48]
 900:	8b000020 	add	x0, x1, x0
 904:	910343e1 	add	x1, sp, #0xd0
 908:	3dc00420 	ldr	q0, [x1, #16]
 90c:	f9002fe0 	str	x0, [sp, #88]
 910:	3d8023e0 	str	q0, [sp, #128]
 914:	3dc023e0 	ldr	q0, [sp, #128]
 918:	f9402fe0 	ldr	x0, [sp, #88]
 91c:	3d800000 	str	q0, [x0]
 920:	d503201f 	nop
 924:	b9404be0 	ldr	w0, [sp, #72]
 928:	531c6c00 	lsl	w0, w0, #4
 92c:	93407c00 	sxtw	x0, w0
 930:	f94017e1 	ldr	x1, [sp, #40]
 934:	8b000020 	add	x0, x1, x0
 938:	910343e1 	add	x1, sp, #0xd0
 93c:	3dc00820 	ldr	q0, [x1, #32]
 940:	f9002be0 	str	x0, [sp, #80]
 944:	3d801fe0 	str	q0, [sp, #112]
 948:	3dc01fe0 	ldr	q0, [sp, #112]
 94c:	f9402be0 	ldr	x0, [sp, #80]
 950:	3d800000 	str	q0, [x0]
 954:	d503201f 	nop
 958:	b9404be0 	ldr	w0, [sp, #72]
 95c:	11000400 	add	w0, w0, #0x1
 960:	b9004be0 	str	w0, [sp, #72]
 964:	b9404be1 	ldr	w1, [sp, #72]
 968:	b9404fe0 	ldr	w0, [sp, #76]
 96c:	6b00003f 	cmp	w1, w0
 970:	54fff66b 	b.lt	83c <rgb_deinterleave_neon+0x50>  // b.tstop
 974:	d503201f 	nop
 978:	90000080 	adrp	x0, 10000 <__FRAME_END__+0xf478>
 97c:	f947f000 	ldr	x0, [x0, #4064]
 980:	f9409fe1 	ldr	x1, [sp, #312]
 984:	f9400002 	ldr	x2, [x0]
 988:	eb020021 	subs	x1, x1, x2
 98c:	d2800002 	mov	x2, #0x0                   	// #0
 990:	54000040 	b.eq	998 <rgb_deinterleave_neon+0x1ac>  // b.none
 994:	97ffff47 	bl	6b0 <__stack_chk_fail@plt>
 998:	a8d47bfd 	ldp	x29, x30, [sp], #320
 99c:	d65f03c0 	ret

00000000000009a0 <main>:
 9a0:	52800000 	mov	w0, #0x0                   	// #0
 9a4:	d65f03c0 	ret

00000000000009a8 <__libc_csu_init>:
 9a8:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
 9ac:	910003fd 	mov	x29, sp
 9b0:	a90153f3 	stp	x19, x20, [sp, #16]
 9b4:	90000094 	adrp	x20, 10000 <__FRAME_END__+0xf478>
 9b8:	9135c294 	add	x20, x20, #0xd70
 9bc:	a9025bf5 	stp	x21, x22, [sp, #32]
 9c0:	90000095 	adrp	x21, 10000 <__FRAME_END__+0xf478>
 9c4:	9135a2b5 	add	x21, x21, #0xd68
 9c8:	cb150294 	sub	x20, x20, x21
 9cc:	2a0003f6 	mov	w22, w0
 9d0:	a90363f7 	stp	x23, x24, [sp, #48]
 9d4:	aa0103f7 	mov	x23, x1
 9d8:	aa0203f8 	mov	x24, x2
 9dc:	97ffff1f 	bl	658 <_init>
 9e0:	eb940fff 	cmp	xzr, x20, asr #3
 9e4:	54000160 	b.eq	a10 <__libc_csu_init+0x68>  // b.none
 9e8:	9343fe94 	asr	x20, x20, #3
 9ec:	d2800013 	mov	x19, #0x0                   	// #0
 9f0:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
 9f4:	aa1803e2 	mov	x2, x24
 9f8:	91000673 	add	x19, x19, #0x1
 9fc:	aa1703e1 	mov	x1, x23
 a00:	2a1603e0 	mov	w0, w22
 a04:	d63f0060 	blr	x3
 a08:	eb13029f 	cmp	x20, x19
 a0c:	54ffff21 	b.ne	9f0 <__libc_csu_init+0x48>  // b.any
 a10:	a94153f3 	ldp	x19, x20, [sp, #16]
 a14:	a9425bf5 	ldp	x21, x22, [sp, #32]
 a18:	a94363f7 	ldp	x23, x24, [sp, #48]
 a1c:	a8c47bfd 	ldp	x29, x30, [sp], #64
 a20:	d65f03c0 	ret
 a24:	d503201f 	nop

0000000000000a28 <__libc_csu_fini>:
 a28:	d65f03c0 	ret

Disassembly of section .fini:

0000000000000a2c <_fini>:
 a2c:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
 a30:	910003fd 	mov	x29, sp
 a34:	a8c17bfd 	ldp	x29, x30, [sp], #16
 a38:	d65f03c0 	ret
