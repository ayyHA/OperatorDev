
array_weighted_sum_in_asm.exe:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000000700 <_init>:
 700:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
 704:	910003fd 	mov	x29, sp
 708:	9400003c 	bl	7f8 <call_weak_fn>
 70c:	a8c17bfd 	ldp	x29, x30, [sp], #16
 710:	d65f03c0 	ret

Disassembly of section .plt:

0000000000000720 <.plt>:
 720:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
 724:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 728:	f947ba11 	ldr	x17, [x16, #3952]
 72c:	913dc210 	add	x16, x16, #0xf70
 730:	d61f0220 	br	x17
 734:	d503201f 	nop
 738:	d503201f 	nop
 73c:	d503201f 	nop

0000000000000740 <__cxa_finalize@plt>:
 740:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 744:	f947be11 	ldr	x17, [x16, #3960]
 748:	913de210 	add	x16, x16, #0xf78
 74c:	d61f0220 	br	x17

0000000000000750 <__libc_start_main@plt>:
 750:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 754:	f947c211 	ldr	x17, [x16, #3968]
 758:	913e0210 	add	x16, x16, #0xf80
 75c:	d61f0220 	br	x17

0000000000000760 <rand@plt>:
 760:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 764:	f947c611 	ldr	x17, [x16, #3976]
 768:	913e2210 	add	x16, x16, #0xf88
 76c:	d61f0220 	br	x17

0000000000000770 <__stack_chk_fail@plt>:
 770:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 774:	f947ca11 	ldr	x17, [x16, #3984]
 778:	913e4210 	add	x16, x16, #0xf90
 77c:	d61f0220 	br	x17

0000000000000780 <__gmon_start__@plt>:
 780:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 784:	f947ce11 	ldr	x17, [x16, #3992]
 788:	913e6210 	add	x16, x16, #0xf98
 78c:	d61f0220 	br	x17

0000000000000790 <abort@plt>:
 790:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 794:	f947d211 	ldr	x17, [x16, #4000]
 798:	913e8210 	add	x16, x16, #0xfa0
 79c:	d61f0220 	br	x17

00000000000007a0 <printf@plt>:
 7a0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 7a4:	f947d611 	ldr	x17, [x16, #4008]
 7a8:	913ea210 	add	x16, x16, #0xfa8
 7ac:	d61f0220 	br	x17

00000000000007b0 <putchar@plt>:
 7b0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xfd68>
 7b4:	f947da11 	ldr	x17, [x16, #4016]
 7b8:	913ec210 	add	x16, x16, #0xfb0
 7bc:	d61f0220 	br	x17

Disassembly of section .text:

00000000000007c0 <_start>:
     7c0:	d280001d 	mov	x29, #0x0                   	// #0
     7c4:	d280001e 	mov	x30, #0x0                   	// #0
     7c8:	aa0003e5 	mov	x5, x0
     7cc:	f94003e1 	ldr	x1, [sp]
     7d0:	910023e2 	add	x2, sp, #0x8
     7d4:	910003e6 	mov	x6, sp
     7d8:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xfd68>
     7dc:	f947f800 	ldr	x0, [x0, #4080]
     7e0:	b0000083 	adrp	x3, 11000 <__FRAME_END__+0xfd68>
     7e4:	f947f463 	ldr	x3, [x3, #4072]
     7e8:	b0000084 	adrp	x4, 11000 <__FRAME_END__+0xfd68>
     7ec:	f947e084 	ldr	x4, [x4, #4032]
     7f0:	97ffffd8 	bl	750 <__libc_start_main@plt>
     7f4:	97ffffe7 	bl	790 <abort@plt>

00000000000007f8 <call_weak_fn>:
     7f8:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xfd68>
     7fc:	f947ec00 	ldr	x0, [x0, #4056]
     800:	b4000040 	cbz	x0, 808 <call_weak_fn+0x10>
     804:	17ffffdf 	b	780 <__gmon_start__@plt>
     808:	d65f03c0 	ret
     80c:	d503201f 	nop

0000000000000810 <deregister_tm_clones>:
     810:	d0000080 	adrp	x0, 12000 <__data_start>
     814:	91004000 	add	x0, x0, #0x10
     818:	d0000081 	adrp	x1, 12000 <__data_start>
     81c:	91004021 	add	x1, x1, #0x10
     820:	eb00003f 	cmp	x1, x0
     824:	540000c0 	b.eq	83c <deregister_tm_clones+0x2c>  // b.none
     828:	b0000081 	adrp	x1, 11000 <__FRAME_END__+0xfd68>
     82c:	f947e421 	ldr	x1, [x1, #4040]
     830:	b4000061 	cbz	x1, 83c <deregister_tm_clones+0x2c>
     834:	aa0103f0 	mov	x16, x1
     838:	d61f0200 	br	x16
     83c:	d65f03c0 	ret

0000000000000840 <register_tm_clones>:
     840:	d0000080 	adrp	x0, 12000 <__data_start>
     844:	91004000 	add	x0, x0, #0x10
     848:	d0000081 	adrp	x1, 12000 <__data_start>
     84c:	91004021 	add	x1, x1, #0x10
     850:	cb000021 	sub	x1, x1, x0
     854:	d37ffc22 	lsr	x2, x1, #63
     858:	8b810c41 	add	x1, x2, x1, asr #3
     85c:	eb8107ff 	cmp	xzr, x1, asr #1
     860:	9341fc21 	asr	x1, x1, #1
     864:	540000c0 	b.eq	87c <register_tm_clones+0x3c>  // b.none
     868:	b0000082 	adrp	x2, 11000 <__FRAME_END__+0xfd68>
     86c:	f947fc42 	ldr	x2, [x2, #4088]
     870:	b4000062 	cbz	x2, 87c <register_tm_clones+0x3c>
     874:	aa0203f0 	mov	x16, x2
     878:	d61f0200 	br	x16
     87c:	d65f03c0 	ret

0000000000000880 <__do_global_dtors_aux>:
     880:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
     884:	910003fd 	mov	x29, sp
     888:	f9000bf3 	str	x19, [sp, #16]
     88c:	d0000093 	adrp	x19, 12000 <__data_start>
     890:	39404260 	ldrb	w0, [x19, #16]
     894:	35000140 	cbnz	w0, 8bc <__do_global_dtors_aux+0x3c>
     898:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xfd68>
     89c:	f947e800 	ldr	x0, [x0, #4048]
     8a0:	b4000080 	cbz	x0, 8b0 <__do_global_dtors_aux+0x30>
     8a4:	d0000080 	adrp	x0, 12000 <__data_start>
     8a8:	f9400400 	ldr	x0, [x0, #8]
     8ac:	97ffffa5 	bl	740 <__cxa_finalize@plt>
     8b0:	97ffffd8 	bl	810 <deregister_tm_clones>
     8b4:	52800020 	mov	w0, #0x1                   	// #1
     8b8:	39004260 	strb	w0, [x19, #16]
     8bc:	f9400bf3 	ldr	x19, [sp, #16]
     8c0:	a8c27bfd 	ldp	x29, x30, [sp], #32
     8c4:	d65f03c0 	ret

00000000000008c8 <frame_dummy>:
     8c8:	17ffffde 	b	840 <register_tm_clones>

00000000000008cc <arrayWeightSumInAsm>:
     8cc:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
     8d0:	910003fd 	mov	x29, sp
     8d4:	f9001fe0 	str	x0, [sp, #56]
     8d8:	bd0037e0 	str	s0, [sp, #52]
     8dc:	f90017e1 	str	x1, [sp, #40]
     8e0:	bd0033e1 	str	s1, [sp, #48]
     8e4:	b90027e2 	str	w2, [sp, #36]
     8e8:	f9000fe3 	str	x3, [sp, #24]
     8ec:	b94027e0 	ldr	w0, [sp, #36]
     8f0:	13027c00 	asr	w0, w0, #2
     8f4:	b9004fe0 	str	w0, [sp, #76]
     8f8:	b9404fe0 	ldr	w0, [sp, #76]
     8fc:	531e7400 	lsl	w0, w0, #2
     900:	b94027e1 	ldr	w1, [sp, #36]
     904:	4b000020 	sub	w0, w1, w0
     908:	b9004be0 	str	w0, [sp, #72]
     90c:	b9404be2 	ldr	w2, [sp, #72]
     910:	b9404fe1 	ldr	w1, [sp, #76]
     914:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     918:	91008000 	add	x0, x0, #0x20
     91c:	97ffffa1 	bl	7a0 <printf@plt>
     920:	b94037e6 	ldr	w6, [sp, #52]
     924:	b94033e7 	ldr	w7, [sp, #48]
     928:	f9401fe3 	ldr	x3, [sp, #56]
     92c:	f94017e2 	ldr	x2, [sp, #40]
     930:	f9400fe1 	ldr	x1, [sp, #24]
     934:	b9404fe0 	ldr	w0, [sp, #76]
     938:	aa0303e5 	mov	x5, x3
     93c:	aa0203e4 	mov	x4, x2
     940:	aa0103e3 	mov	x3, x1
     944:	2a0003e2 	mov	w2, w0
     948:	aa0603e0 	mov	x0, x6
     94c:	4e040c00 	dup	v0.4s, w0
     950:	aa0703e1 	mov	x1, x7
     954:	4e040c21 	dup	v1.4s, w1
     958:	f98040a0 	prfm	pldl1keep, [x5, #128]
     95c:	4cdf78a2 	ld1	{v2.4s}, [x5], #16
     960:	f9804080 	prfm	pldl1keep, [x4, #128]
     964:	4cdf7883 	ld1	{v3.4s}, [x4], #16
     968:	6e22dc04 	fmul	v4.4s, v0.4s, v2.4s
     96c:	6e23dc25 	fmul	v5.4s, v1.4s, v3.4s
     970:	4e25d484 	fadd	v4.4s, v4.4s, v5.4s
     974:	4c9f7864 	st1	{v4.4s}, [x3], #16
     978:	f1000442 	subs	x2, x2, #0x1
     97c:	54fffeec 	b.gt	958 <arrayWeightSumInAsm+0x8c>
     980:	f9001fe5 	str	x5, [sp, #56]
     984:	f90017e4 	str	x4, [sp, #40]
     988:	f9000fe3 	str	x3, [sp, #24]
     98c:	b9004fe2 	str	w2, [sp, #76]
     990:	14000018 	b	9f0 <arrayWeightSumInAsm+0x124>
     994:	f9401fe0 	ldr	x0, [sp, #56]
     998:	bd400001 	ldr	s1, [x0]
     99c:	bd4037e0 	ldr	s0, [sp, #52]
     9a0:	1e200821 	fmul	s1, s1, s0
     9a4:	f94017e0 	ldr	x0, [sp, #40]
     9a8:	bd400002 	ldr	s2, [x0]
     9ac:	bd4033e0 	ldr	s0, [sp, #48]
     9b0:	1e200840 	fmul	s0, s2, s0
     9b4:	1e202820 	fadd	s0, s1, s0
     9b8:	f9400fe0 	ldr	x0, [sp, #24]
     9bc:	bd000000 	str	s0, [x0]
     9c0:	f9400fe0 	ldr	x0, [sp, #24]
     9c4:	91001000 	add	x0, x0, #0x4
     9c8:	f9000fe0 	str	x0, [sp, #24]
     9cc:	f9401fe0 	ldr	x0, [sp, #56]
     9d0:	91001000 	add	x0, x0, #0x4
     9d4:	f9001fe0 	str	x0, [sp, #56]
     9d8:	f94017e0 	ldr	x0, [sp, #40]
     9dc:	91001000 	add	x0, x0, #0x4
     9e0:	f90017e0 	str	x0, [sp, #40]
     9e4:	b9404be0 	ldr	w0, [sp, #72]
     9e8:	51000400 	sub	w0, w0, #0x1
     9ec:	b9004be0 	str	w0, [sp, #72]
     9f0:	b9404be0 	ldr	w0, [sp, #72]
     9f4:	7100001f 	cmp	w0, #0x0
     9f8:	54fffcec 	b.gt	994 <arrayWeightSumInAsm+0xc8>
     9fc:	d503201f 	nop
     a00:	d503201f 	nop
     a04:	a8c57bfd 	ldp	x29, x30, [sp], #80
     a08:	d65f03c0 	ret

0000000000000a0c <arrayWeightSumNeon>:
     a0c:	d10483ff 	sub	sp, sp, #0x120
     a10:	f90017e0 	str	x0, [sp, #40]
     a14:	bd0027e0 	str	s0, [sp, #36]
     a18:	f9000fe1 	str	x1, [sp, #24]
     a1c:	bd0023e1 	str	s1, [sp, #32]
     a20:	b90017e2 	str	w2, [sp, #20]
     a24:	f90007e3 	str	x3, [sp, #8]
     a28:	b94017e0 	ldr	w0, [sp, #20]
     a2c:	13027c00 	asr	w0, w0, #2
     a30:	b9003fe0 	str	w0, [sp, #60]
     a34:	b94017e1 	ldr	w1, [sp, #20]
     a38:	b9403fe0 	ldr	w0, [sp, #60]
     a3c:	4b000020 	sub	w0, w1, w0
     a40:	531e7400 	lsl	w0, w0, #2
     a44:	b90037e0 	str	w0, [sp, #52]
     a48:	bd4027e0 	ldr	s0, [sp, #36]
     a4c:	bd0047e0 	str	s0, [sp, #68]
     a50:	bd4047e0 	ldr	s0, [sp, #68]
     a54:	4e040400 	dup	v0.4s, v0.s[0]
     a58:	3d801be0 	str	q0, [sp, #96]
     a5c:	bd4023e0 	ldr	s0, [sp, #32]
     a60:	bd0043e0 	str	s0, [sp, #64]
     a64:	bd4043e0 	ldr	s0, [sp, #64]
     a68:	4e040400 	dup	v0.4s, v0.s[0]
     a6c:	3d801fe0 	str	q0, [sp, #112]
     a70:	b9003bff 	str	wzr, [sp, #56]
     a74:	14000037 	b	b50 <arrayWeightSumNeon+0x144>
     a78:	f94017e0 	ldr	x0, [sp, #40]
     a7c:	f9002fe0 	str	x0, [sp, #88]
     a80:	f9402fe0 	ldr	x0, [sp, #88]
     a84:	3dc00000 	ldr	q0, [x0]
     a88:	3d8023e0 	str	q0, [sp, #128]
     a8c:	f9400fe0 	ldr	x0, [sp, #24]
     a90:	f9002be0 	str	x0, [sp, #80]
     a94:	f9402be0 	ldr	x0, [sp, #80]
     a98:	3dc00000 	ldr	q0, [x0]
     a9c:	3d8027e0 	str	q0, [sp, #144]
     aa0:	3dc023e0 	ldr	q0, [sp, #128]
     aa4:	3d8043e0 	str	q0, [sp, #256]
     aa8:	3dc01be0 	ldr	q0, [sp, #96]
     aac:	3d8047e0 	str	q0, [sp, #272]
     ab0:	3dc043e1 	ldr	q1, [sp, #256]
     ab4:	3dc047e0 	ldr	q0, [sp, #272]
     ab8:	6e20dc20 	fmul	v0.4s, v1.4s, v0.4s
     abc:	3d8023e0 	str	q0, [sp, #128]
     ac0:	3dc027e0 	ldr	q0, [sp, #144]
     ac4:	3d803be0 	str	q0, [sp, #224]
     ac8:	3dc01fe0 	ldr	q0, [sp, #112]
     acc:	3d803fe0 	str	q0, [sp, #240]
     ad0:	3dc03be1 	ldr	q1, [sp, #224]
     ad4:	3dc03fe0 	ldr	q0, [sp, #240]
     ad8:	6e20dc20 	fmul	v0.4s, v1.4s, v0.4s
     adc:	3d8027e0 	str	q0, [sp, #144]
     ae0:	3dc023e0 	ldr	q0, [sp, #128]
     ae4:	3d8033e0 	str	q0, [sp, #192]
     ae8:	3dc027e0 	ldr	q0, [sp, #144]
     aec:	3d8037e0 	str	q0, [sp, #208]
     af0:	3dc033e1 	ldr	q1, [sp, #192]
     af4:	3dc037e0 	ldr	q0, [sp, #208]
     af8:	4e20d420 	fadd	v0.4s, v1.4s, v0.4s
     afc:	3d802be0 	str	q0, [sp, #160]
     b00:	f94007e0 	ldr	x0, [sp, #8]
     b04:	f90027e0 	str	x0, [sp, #72]
     b08:	3dc02be0 	ldr	q0, [sp, #160]
     b0c:	3d802fe0 	str	q0, [sp, #176]
     b10:	f94027e0 	ldr	x0, [sp, #72]
     b14:	3dc02fe0 	ldr	q0, [sp, #176]
     b18:	3d800000 	str	q0, [x0]
     b1c:	d503201f 	nop
     b20:	f94017e0 	ldr	x0, [sp, #40]
     b24:	91004000 	add	x0, x0, #0x10
     b28:	f90017e0 	str	x0, [sp, #40]
     b2c:	f9400fe0 	ldr	x0, [sp, #24]
     b30:	91004000 	add	x0, x0, #0x10
     b34:	f9000fe0 	str	x0, [sp, #24]
     b38:	f94007e0 	ldr	x0, [sp, #8]
     b3c:	91004000 	add	x0, x0, #0x10
     b40:	f90007e0 	str	x0, [sp, #8]
     b44:	b9403be0 	ldr	w0, [sp, #56]
     b48:	11000400 	add	w0, w0, #0x1
     b4c:	b9003be0 	str	w0, [sp, #56]
     b50:	b9403be1 	ldr	w1, [sp, #56]
     b54:	b9403fe0 	ldr	w0, [sp, #60]
     b58:	6b00003f 	cmp	w1, w0
     b5c:	54fff8eb 	b.lt	a78 <arrayWeightSumNeon+0x6c>  // b.tstop
     b60:	14000018 	b	bc0 <arrayWeightSumNeon+0x1b4>
     b64:	f94017e0 	ldr	x0, [sp, #40]
     b68:	bd400001 	ldr	s1, [x0]
     b6c:	bd4027e0 	ldr	s0, [sp, #36]
     b70:	1e200821 	fmul	s1, s1, s0
     b74:	f9400fe0 	ldr	x0, [sp, #24]
     b78:	bd400002 	ldr	s2, [x0]
     b7c:	bd4023e0 	ldr	s0, [sp, #32]
     b80:	1e200840 	fmul	s0, s2, s0
     b84:	1e202820 	fadd	s0, s1, s0
     b88:	f94007e0 	ldr	x0, [sp, #8]
     b8c:	bd000000 	str	s0, [x0]
     b90:	f94007e0 	ldr	x0, [sp, #8]
     b94:	91001000 	add	x0, x0, #0x4
     b98:	f90007e0 	str	x0, [sp, #8]
     b9c:	f94017e0 	ldr	x0, [sp, #40]
     ba0:	91001000 	add	x0, x0, #0x4
     ba4:	f90017e0 	str	x0, [sp, #40]
     ba8:	f9400fe0 	ldr	x0, [sp, #24]
     bac:	91001000 	add	x0, x0, #0x4
     bb0:	f9000fe0 	str	x0, [sp, #24]
     bb4:	b94037e0 	ldr	w0, [sp, #52]
     bb8:	51000400 	sub	w0, w0, #0x1
     bbc:	b90037e0 	str	w0, [sp, #52]
     bc0:	b94037e0 	ldr	w0, [sp, #52]
     bc4:	7100001f 	cmp	w0, #0x0
     bc8:	54fffcec 	b.gt	b64 <arrayWeightSumNeon+0x158>
     bcc:	d503201f 	nop
     bd0:	d503201f 	nop
     bd4:	910483ff 	add	sp, sp, #0x120
     bd8:	d65f03c0 	ret

0000000000000bdc <arrayWeightSum>:
     bdc:	d10103ff 	sub	sp, sp, #0x40
     be0:	f90017e0 	str	x0, [sp, #40]
     be4:	bd0027e0 	str	s0, [sp, #36]
     be8:	f9000fe1 	str	x1, [sp, #24]
     bec:	bd0023e1 	str	s1, [sp, #32]
     bf0:	b90017e2 	str	w2, [sp, #20]
     bf4:	f90007e3 	str	x3, [sp, #8]
     bf8:	b9003fff 	str	wzr, [sp, #60]
     bfc:	14000018 	b	c5c <arrayWeightSum+0x80>
     c00:	b9803fe0 	ldrsw	x0, [sp, #60]
     c04:	d37ef400 	lsl	x0, x0, #2
     c08:	f94017e1 	ldr	x1, [sp, #40]
     c0c:	8b000020 	add	x0, x1, x0
     c10:	bd400001 	ldr	s1, [x0]
     c14:	bd4027e0 	ldr	s0, [sp, #36]
     c18:	1e200821 	fmul	s1, s1, s0
     c1c:	b9803fe0 	ldrsw	x0, [sp, #60]
     c20:	d37ef400 	lsl	x0, x0, #2
     c24:	f9400fe1 	ldr	x1, [sp, #24]
     c28:	8b000020 	add	x0, x1, x0
     c2c:	bd400002 	ldr	s2, [x0]
     c30:	bd4023e0 	ldr	s0, [sp, #32]
     c34:	1e200840 	fmul	s0, s2, s0
     c38:	b9803fe0 	ldrsw	x0, [sp, #60]
     c3c:	d37ef400 	lsl	x0, x0, #2
     c40:	f94007e1 	ldr	x1, [sp, #8]
     c44:	8b000020 	add	x0, x1, x0
     c48:	1e202820 	fadd	s0, s1, s0
     c4c:	bd000000 	str	s0, [x0]
     c50:	b9403fe0 	ldr	w0, [sp, #60]
     c54:	11000400 	add	w0, w0, #0x1
     c58:	b9003fe0 	str	w0, [sp, #60]
     c5c:	b9403fe1 	ldr	w1, [sp, #60]
     c60:	b94017e0 	ldr	w0, [sp, #20]
     c64:	6b00003f 	cmp	w1, w0
     c68:	54fffccb 	b.lt	c00 <arrayWeightSum+0x24>  // b.tstop
     c6c:	d503201f 	nop
     c70:	d503201f 	nop
     c74:	910103ff 	add	sp, sp, #0x40
     c78:	d65f03c0 	ret

0000000000000c7c <generate_array>:
     c7c:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
     c80:	910003fd 	mov	x29, sp
     c84:	f9000fe0 	str	x0, [sp, #24]
     c88:	b90017e1 	str	w1, [sp, #20]
     c8c:	b9002fff 	str	wzr, [sp, #44]
     c90:	1400000d 	b	cc4 <generate_array+0x48>
     c94:	97fffeb3 	bl	760 <rand@plt>
     c98:	1e220001 	scvtf	s1, w0
     c9c:	b9802fe0 	ldrsw	x0, [sp, #44]
     ca0:	d37ef400 	lsl	x0, x0, #2
     ca4:	f9400fe1 	ldr	x1, [sp, #24]
     ca8:	8b000020 	add	x0, x1, x0
     cac:	0f0265e0 	movi	v0.2s, #0x4f, lsl #24
     cb0:	1e201820 	fdiv	s0, s1, s0
     cb4:	bd000000 	str	s0, [x0]
     cb8:	b9402fe0 	ldr	w0, [sp, #44]
     cbc:	11000400 	add	w0, w0, #0x1
     cc0:	b9002fe0 	str	w0, [sp, #44]
     cc4:	b9402fe1 	ldr	w1, [sp, #44]
     cc8:	b94017e0 	ldr	w0, [sp, #20]
     ccc:	6b00003f 	cmp	w1, w0
     cd0:	54fffe2b 	b.lt	c94 <generate_array+0x18>  // b.tstop
     cd4:	d503201f 	nop
     cd8:	d503201f 	nop
     cdc:	a8c37bfd 	ldp	x29, x30, [sp], #48
     ce0:	d65f03c0 	ret

0000000000000ce4 <print_array>:
     ce4:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
     ce8:	910003fd 	mov	x29, sp
     cec:	f9000fe0 	str	x0, [sp, #24]
     cf0:	b90017e1 	str	w1, [sp, #20]
     cf4:	b9002fff 	str	wzr, [sp, #44]
     cf8:	1400000d 	b	d2c <print_array+0x48>
     cfc:	b9802fe0 	ldrsw	x0, [sp, #44]
     d00:	d37ef400 	lsl	x0, x0, #2
     d04:	f9400fe1 	ldr	x1, [sp, #24]
     d08:	8b000020 	add	x0, x1, x0
     d0c:	bd400000 	ldr	s0, [x0]
     d10:	1e22c000 	fcvt	d0, s0
     d14:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     d18:	91010000 	add	x0, x0, #0x40
     d1c:	97fffea1 	bl	7a0 <printf@plt>
     d20:	b9402fe0 	ldr	w0, [sp, #44]
     d24:	11000400 	add	w0, w0, #0x1
     d28:	b9002fe0 	str	w0, [sp, #44]
     d2c:	b9402fe1 	ldr	w1, [sp, #44]
     d30:	b94017e0 	ldr	w0, [sp, #20]
     d34:	6b00003f 	cmp	w1, w0
     d38:	54fffe2b 	b.lt	cfc <print_array+0x18>  // b.tstop
     d3c:	52800140 	mov	w0, #0xa                   	// #10
     d40:	97fffe9c 	bl	7b0 <putchar@plt>
     d44:	d503201f 	nop
     d48:	a8c37bfd 	ldp	x29, x30, [sp], #48
     d4c:	d65f03c0 	ret

0000000000000d50 <compare_array>:
     d50:	d100c3ff 	sub	sp, sp, #0x30
     d54:	f9000fe0 	str	x0, [sp, #24]
     d58:	f9000be1 	str	x1, [sp, #16]
     d5c:	b9000fe2 	str	w2, [sp, #12]
     d60:	b9002fff 	str	wzr, [sp, #44]
     d64:	14000017 	b	dc0 <compare_array+0x70>
     d68:	b9802fe0 	ldrsw	x0, [sp, #44]
     d6c:	d37ef400 	lsl	x0, x0, #2
     d70:	f9400fe1 	ldr	x1, [sp, #24]
     d74:	8b000020 	add	x0, x1, x0
     d78:	bd400001 	ldr	s1, [x0]
     d7c:	b9802fe0 	ldrsw	x0, [sp, #44]
     d80:	d37ef400 	lsl	x0, x0, #2
     d84:	f9400be1 	ldr	x1, [sp, #16]
     d88:	8b000020 	add	x0, x1, x0
     d8c:	bd400000 	ldr	s0, [x0]
     d90:	1e203820 	fsub	s0, s1, s0
     d94:	1e20c000 	fabs	s0, s0
     d98:	1e22c000 	fcvt	d0, s0
     d9c:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     da0:	fd404401 	ldr	d1, [x0, #136]
     da4:	1e612010 	fcmpe	d0, d1
     da8:	5400006d 	b.le	db4 <compare_array+0x64>
     dac:	52800000 	mov	w0, #0x0                   	// #0
     db0:	14000009 	b	dd4 <compare_array+0x84>
     db4:	b9402fe0 	ldr	w0, [sp, #44]
     db8:	11000400 	add	w0, w0, #0x1
     dbc:	b9002fe0 	str	w0, [sp, #44]
     dc0:	b9402fe1 	ldr	w1, [sp, #44]
     dc4:	b9400fe0 	ldr	w0, [sp, #12]
     dc8:	6b00003f 	cmp	w1, w0
     dcc:	54fffceb 	b.lt	d68 <compare_array+0x18>  // b.tstop
     dd0:	52800020 	mov	w0, #0x1                   	// #1
     dd4:	9100c3ff 	add	sp, sp, #0x30
     dd8:	d65f03c0 	ret

0000000000000ddc <main>:
     ddc:	a9a97bfd 	stp	x29, x30, [sp, #-368]!
     de0:	910003fd 	mov	x29, sp
     de4:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xfd68>
     de8:	f947f000 	ldr	x0, [x0, #4064]
     dec:	f9400001 	ldr	x1, [x0]
     df0:	f900b7e1 	str	x1, [sp, #360]
     df4:	d2800001 	mov	x1, #0x0                   	// #0
     df8:	9100a3e0 	add	x0, sp, #0x28
     dfc:	52800141 	mov	w1, #0xa                   	// #10
     e00:	97ffff9f 	bl	c7c <generate_array>
     e04:	9101a3e0 	add	x0, sp, #0x68
     e08:	52800141 	mov	w1, #0xa                   	// #10
     e0c:	97ffff9c 	bl	c7c <generate_array>
     e10:	9100a3e0 	add	x0, sp, #0x28
     e14:	52800141 	mov	w1, #0xa                   	// #10
     e18:	97ffffb3 	bl	ce4 <print_array>
     e1c:	9101a3e0 	add	x0, sp, #0x68
     e20:	52800141 	mov	w1, #0xa                   	// #10
     e24:	97ffffb0 	bl	ce4 <print_array>
     e28:	9103a3e2 	add	x2, sp, #0xe8
     e2c:	9101a3e1 	add	x1, sp, #0x68
     e30:	9100a3e0 	add	x0, sp, #0x28
     e34:	aa0203e3 	mov	x3, x2
     e38:	52800142 	mov	w2, #0xa                   	// #10
     e3c:	529999a4 	mov	w4, #0xcccd                	// #52429
     e40:	72a7d984 	movk	w4, #0x3ecc, lsl #16
     e44:	1e270081 	fmov	s1, w4
     e48:	1e2c1000 	fmov	s0, #5.000000000000000000e-01
     e4c:	97ffff64 	bl	bdc <arrayWeightSum>
     e50:	9103a3e0 	add	x0, sp, #0xe8
     e54:	52800141 	mov	w1, #0xa                   	// #10
     e58:	97ffffa3 	bl	ce4 <print_array>
     e5c:	9104a3e2 	add	x2, sp, #0x128
     e60:	9101a3e1 	add	x1, sp, #0x68
     e64:	9100a3e0 	add	x0, sp, #0x28
     e68:	aa0203e3 	mov	x3, x2
     e6c:	52800142 	mov	w2, #0xa                   	// #10
     e70:	529999a4 	mov	w4, #0xcccd                	// #52429
     e74:	72a7d984 	movk	w4, #0x3ecc, lsl #16
     e78:	1e270081 	fmov	s1, w4
     e7c:	1e2c1000 	fmov	s0, #5.000000000000000000e-01
     e80:	97fffee3 	bl	a0c <arrayWeightSumNeon>
     e84:	9104a3e0 	add	x0, sp, #0x128
     e88:	52800141 	mov	w1, #0xa                   	// #10
     e8c:	97ffff96 	bl	ce4 <print_array>
     e90:	9102a3e2 	add	x2, sp, #0xa8
     e94:	9101a3e1 	add	x1, sp, #0x68
     e98:	9100a3e0 	add	x0, sp, #0x28
     e9c:	aa0203e3 	mov	x3, x2
     ea0:	52800142 	mov	w2, #0xa                   	// #10
     ea4:	529999a4 	mov	w4, #0xcccd                	// #52429
     ea8:	72a7d984 	movk	w4, #0x3ecc, lsl #16
     eac:	1e270081 	fmov	s1, w4
     eb0:	1e2c1000 	fmov	s0, #5.000000000000000000e-01
     eb4:	97fffe86 	bl	8cc <arrayWeightSumInAsm>
     eb8:	9102a3e0 	add	x0, sp, #0xa8
     ebc:	52800141 	mov	w1, #0xa                   	// #10
     ec0:	97ffff89 	bl	ce4 <print_array>
     ec4:	9104a3e1 	add	x1, sp, #0x128
     ec8:	9103a3e0 	add	x0, sp, #0xe8
     ecc:	52800142 	mov	w2, #0xa                   	// #10
     ed0:	97ffffa0 	bl	d50 <compare_array>
     ed4:	12001c00 	and	w0, w0, #0xff
     ed8:	7100001f 	cmp	w0, #0x0
     edc:	54000080 	b.eq	eec <main+0x110>  // b.none
     ee0:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     ee4:	91012000 	add	x0, x0, #0x48
     ee8:	14000003 	b	ef4 <main+0x118>
     eec:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     ef0:	91014000 	add	x0, x0, #0x50
     ef4:	f9000fe0 	str	x0, [sp, #24]
     ef8:	f9400fe1 	ldr	x1, [sp, #24]
     efc:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     f00:	91016000 	add	x0, x0, #0x58
     f04:	97fffe27 	bl	7a0 <printf@plt>
     f08:	9102a3e1 	add	x1, sp, #0xa8
     f0c:	9103a3e0 	add	x0, sp, #0xe8
     f10:	52800142 	mov	w2, #0xa                   	// #10
     f14:	97ffff8f 	bl	d50 <compare_array>
     f18:	12001c00 	and	w0, w0, #0xff
     f1c:	7100001f 	cmp	w0, #0x0
     f20:	54000080 	b.eq	f30 <main+0x154>  // b.none
     f24:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     f28:	91012000 	add	x0, x0, #0x48
     f2c:	14000003 	b	f38 <main+0x15c>
     f30:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     f34:	91014000 	add	x0, x0, #0x50
     f38:	f90013e0 	str	x0, [sp, #32]
     f3c:	f94013e1 	ldr	x1, [sp, #32]
     f40:	b0000000 	adrp	x0, 1000 <__libc_csu_fini>
     f44:	9101c000 	add	x0, x0, #0x70
     f48:	97fffe16 	bl	7a0 <printf@plt>
     f4c:	52800000 	mov	w0, #0x0                   	// #0
     f50:	2a0003e1 	mov	w1, w0
     f54:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xfd68>
     f58:	f947f000 	ldr	x0, [x0, #4064]
     f5c:	f940b7e2 	ldr	x2, [sp, #360]
     f60:	f9400003 	ldr	x3, [x0]
     f64:	eb030042 	subs	x2, x2, x3
     f68:	d2800003 	mov	x3, #0x0                   	// #0
     f6c:	54000040 	b.eq	f74 <main+0x198>  // b.none
     f70:	97fffe00 	bl	770 <__stack_chk_fail@plt>
     f74:	2a0103e0 	mov	w0, w1
     f78:	a8d77bfd 	ldp	x29, x30, [sp], #368
     f7c:	d65f03c0 	ret

0000000000000f80 <__libc_csu_init>:
     f80:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
     f84:	910003fd 	mov	x29, sp
     f88:	a90153f3 	stp	x19, x20, [sp, #16]
     f8c:	b0000094 	adrp	x20, 11000 <__FRAME_END__+0xfd68>
     f90:	91356294 	add	x20, x20, #0xd58
     f94:	a9025bf5 	stp	x21, x22, [sp, #32]
     f98:	b0000095 	adrp	x21, 11000 <__FRAME_END__+0xfd68>
     f9c:	913542b5 	add	x21, x21, #0xd50
     fa0:	cb150294 	sub	x20, x20, x21
     fa4:	2a0003f6 	mov	w22, w0
     fa8:	a90363f7 	stp	x23, x24, [sp, #48]
     fac:	aa0103f7 	mov	x23, x1
     fb0:	aa0203f8 	mov	x24, x2
     fb4:	97fffdd3 	bl	700 <_init>
     fb8:	eb940fff 	cmp	xzr, x20, asr #3
     fbc:	54000160 	b.eq	fe8 <__libc_csu_init+0x68>  // b.none
     fc0:	9343fe94 	asr	x20, x20, #3
     fc4:	d2800013 	mov	x19, #0x0                   	// #0
     fc8:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
     fcc:	aa1803e2 	mov	x2, x24
     fd0:	91000673 	add	x19, x19, #0x1
     fd4:	aa1703e1 	mov	x1, x23
     fd8:	2a1603e0 	mov	w0, w22
     fdc:	d63f0060 	blr	x3
     fe0:	eb13029f 	cmp	x20, x19
     fe4:	54ffff21 	b.ne	fc8 <__libc_csu_init+0x48>  // b.any
     fe8:	a94153f3 	ldp	x19, x20, [sp, #16]
     fec:	a9425bf5 	ldp	x21, x22, [sp, #32]
     ff0:	a94363f7 	ldp	x23, x24, [sp, #48]
     ff4:	a8c47bfd 	ldp	x29, x30, [sp], #64
     ff8:	d65f03c0 	ret
     ffc:	d503201f 	nop

0000000000001000 <__libc_csu_fini>:
    1000:	d65f03c0 	ret

Disassembly of section .fini:

0000000000001004 <_fini>:
    1004:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1008:	910003fd 	mov	x29, sp
    100c:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1010:	d65f03c0 	ret
