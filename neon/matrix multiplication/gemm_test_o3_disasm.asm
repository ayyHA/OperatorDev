
gemm_test_o3.exe:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000000738 <_init>:
 738:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
 73c:	910003fd 	mov	x29, sp
 740:	9400003e 	bl	838 <call_weak_fn>
 744:	a8c17bfd 	ldp	x29, x30, [sp], #16
 748:	d65f03c0 	ret

Disassembly of section .plt:

0000000000000750 <.plt>:
 750:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
 754:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 758:	f947b611 	ldr	x17, [x16, #3944]
 75c:	913da210 	add	x16, x16, #0xf68
 760:	d61f0220 	br	x17
 764:	d503201f 	nop
 768:	d503201f 	nop
 76c:	d503201f 	nop

0000000000000770 <__cxa_finalize@plt>:
 770:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 774:	f947ba11 	ldr	x17, [x16, #3952]
 778:	913dc210 	add	x16, x16, #0xf70
 77c:	d61f0220 	br	x17

0000000000000780 <__libc_start_main@plt>:
 780:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 784:	f947be11 	ldr	x17, [x16, #3960]
 788:	913de210 	add	x16, x16, #0xf78
 78c:	d61f0220 	br	x17

0000000000000790 <rand@plt>:
 790:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 794:	f947c211 	ldr	x17, [x16, #3968]
 798:	913e0210 	add	x16, x16, #0xf80
 79c:	d61f0220 	br	x17

00000000000007a0 <__stack_chk_fail@plt>:
 7a0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 7a4:	f947c611 	ldr	x17, [x16, #3976]
 7a8:	913e2210 	add	x16, x16, #0xf88
 7ac:	d61f0220 	br	x17

00000000000007b0 <__gmon_start__@plt>:
 7b0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 7b4:	f947ca11 	ldr	x17, [x16, #3984]
 7b8:	913e4210 	add	x16, x16, #0xf90
 7bc:	d61f0220 	br	x17

00000000000007c0 <abort@plt>:
 7c0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 7c4:	f947ce11 	ldr	x17, [x16, #3992]
 7c8:	913e6210 	add	x16, x16, #0xf98
 7cc:	d61f0220 	br	x17

00000000000007d0 <puts@plt>:
 7d0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 7d4:	f947d211 	ldr	x17, [x16, #4000]
 7d8:	913e8210 	add	x16, x16, #0xfa0
 7dc:	d61f0220 	br	x17

00000000000007e0 <printf@plt>:
 7e0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 7e4:	f947d611 	ldr	x17, [x16, #4008]
 7e8:	913ea210 	add	x16, x16, #0xfa8
 7ec:	d61f0220 	br	x17

00000000000007f0 <putchar@plt>:
 7f0:	b0000090 	adrp	x16, 11000 <__FRAME_END__+0xf340>
 7f4:	f947da11 	ldr	x17, [x16, #4016]
 7f8:	913ec210 	add	x16, x16, #0xfb0
 7fc:	d61f0220 	br	x17

Disassembly of section .text:

0000000000000800 <_start>:
     800:	d280001d 	mov	x29, #0x0                   	// #0
     804:	d280001e 	mov	x30, #0x0                   	// #0
     808:	aa0003e5 	mov	x5, x0
     80c:	f94003e1 	ldr	x1, [sp]
     810:	910023e2 	add	x2, sp, #0x8
     814:	910003e6 	mov	x6, sp
     818:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
     81c:	f947f800 	ldr	x0, [x0, #4080]
     820:	b0000083 	adrp	x3, 11000 <__FRAME_END__+0xf340>
     824:	f947f463 	ldr	x3, [x3, #4072]
     828:	b0000084 	adrp	x4, 11000 <__FRAME_END__+0xf340>
     82c:	f947e084 	ldr	x4, [x4, #4032]
     830:	97ffffd4 	bl	780 <__libc_start_main@plt>
     834:	97ffffe3 	bl	7c0 <abort@plt>

0000000000000838 <call_weak_fn>:
     838:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
     83c:	f947ec00 	ldr	x0, [x0, #4056]
     840:	b4000040 	cbz	x0, 848 <call_weak_fn+0x10>
     844:	17ffffdb 	b	7b0 <__gmon_start__@plt>
     848:	d65f03c0 	ret
     84c:	d503201f 	nop

0000000000000850 <deregister_tm_clones>:
     850:	d0000080 	adrp	x0, 12000 <__data_start>
     854:	91004000 	add	x0, x0, #0x10
     858:	d0000081 	adrp	x1, 12000 <__data_start>
     85c:	91004021 	add	x1, x1, #0x10
     860:	eb00003f 	cmp	x1, x0
     864:	540000c0 	b.eq	87c <deregister_tm_clones+0x2c>  // b.none
     868:	b0000081 	adrp	x1, 11000 <__FRAME_END__+0xf340>
     86c:	f947e421 	ldr	x1, [x1, #4040]
     870:	b4000061 	cbz	x1, 87c <deregister_tm_clones+0x2c>
     874:	aa0103f0 	mov	x16, x1
     878:	d61f0200 	br	x16
     87c:	d65f03c0 	ret

0000000000000880 <register_tm_clones>:
     880:	d0000080 	adrp	x0, 12000 <__data_start>
     884:	91004000 	add	x0, x0, #0x10
     888:	d0000081 	adrp	x1, 12000 <__data_start>
     88c:	91004021 	add	x1, x1, #0x10
     890:	cb000021 	sub	x1, x1, x0
     894:	d37ffc22 	lsr	x2, x1, #63
     898:	8b810c41 	add	x1, x2, x1, asr #3
     89c:	eb8107ff 	cmp	xzr, x1, asr #1
     8a0:	9341fc21 	asr	x1, x1, #1
     8a4:	540000c0 	b.eq	8bc <register_tm_clones+0x3c>  // b.none
     8a8:	b0000082 	adrp	x2, 11000 <__FRAME_END__+0xf340>
     8ac:	f947fc42 	ldr	x2, [x2, #4088]
     8b0:	b4000062 	cbz	x2, 8bc <register_tm_clones+0x3c>
     8b4:	aa0203f0 	mov	x16, x2
     8b8:	d61f0200 	br	x16
     8bc:	d65f03c0 	ret

00000000000008c0 <__do_global_dtors_aux>:
     8c0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
     8c4:	910003fd 	mov	x29, sp
     8c8:	f9000bf3 	str	x19, [sp, #16]
     8cc:	d0000093 	adrp	x19, 12000 <__data_start>
     8d0:	39404260 	ldrb	w0, [x19, #16]
     8d4:	35000140 	cbnz	w0, 8fc <__do_global_dtors_aux+0x3c>
     8d8:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
     8dc:	f947e800 	ldr	x0, [x0, #4048]
     8e0:	b4000080 	cbz	x0, 8f0 <__do_global_dtors_aux+0x30>
     8e4:	d0000080 	adrp	x0, 12000 <__data_start>
     8e8:	f9400400 	ldr	x0, [x0, #8]
     8ec:	97ffffa1 	bl	770 <__cxa_finalize@plt>
     8f0:	97ffffd8 	bl	850 <deregister_tm_clones>
     8f4:	52800020 	mov	w0, #0x1                   	// #1
     8f8:	39004260 	strb	w0, [x19, #16]
     8fc:	f9400bf3 	ldr	x19, [sp, #16]
     900:	a8c27bfd 	ldp	x29, x30, [sp], #32
     904:	d65f03c0 	ret

0000000000000908 <frame_dummy>:
     908:	17ffffde 	b	880 <register_tm_clones>

000000000000090c <matrix_multiply_c>:
     90c:	d10103ff 	sub	sp, sp, #0x40
     910:	f90017e0 	str	x0, [sp, #40]
     914:	f90013e1 	str	x1, [sp, #32]
     918:	f9000fe2 	str	x2, [sp, #24]
     91c:	b90017e3 	str	w3, [sp, #20]
     920:	b90013e4 	str	w4, [sp, #16]
     924:	b9000fe5 	str	w5, [sp, #12]
     928:	b90037ff 	str	wzr, [sp, #52]
     92c:	1400004a 	b	a54 <matrix_multiply_c+0x148>
     930:	b9003bff 	str	wzr, [sp, #56]
     934:	14000041 	b	a38 <matrix_multiply_c+0x12c>
     938:	b9403be1 	ldr	w1, [sp, #56]
     93c:	b94017e0 	ldr	w0, [sp, #20]
     940:	1b007c21 	mul	w1, w1, w0
     944:	b94037e0 	ldr	w0, [sp, #52]
     948:	0b000020 	add	w0, w1, w0
     94c:	2a0003e0 	mov	w0, w0
     950:	d37ef400 	lsl	x0, x0, #2
     954:	f9400fe1 	ldr	x1, [sp, #24]
     958:	8b000020 	add	x0, x1, x0
     95c:	b900001f 	str	wzr, [x0]
     960:	b9003fff 	str	wzr, [sp, #60]
     964:	1400002e 	b	a1c <matrix_multiply_c+0x110>
     968:	b9403be1 	ldr	w1, [sp, #56]
     96c:	b94017e0 	ldr	w0, [sp, #20]
     970:	1b007c21 	mul	w1, w1, w0
     974:	b94037e0 	ldr	w0, [sp, #52]
     978:	0b000020 	add	w0, w1, w0
     97c:	2a0003e0 	mov	w0, w0
     980:	d37ef400 	lsl	x0, x0, #2
     984:	f9400fe1 	ldr	x1, [sp, #24]
     988:	8b000020 	add	x0, x1, x0
     98c:	bd400001 	ldr	s1, [x0]
     990:	b9403fe1 	ldr	w1, [sp, #60]
     994:	b94017e0 	ldr	w0, [sp, #20]
     998:	1b007c21 	mul	w1, w1, w0
     99c:	b94037e0 	ldr	w0, [sp, #52]
     9a0:	0b000020 	add	w0, w1, w0
     9a4:	2a0003e0 	mov	w0, w0
     9a8:	d37ef400 	lsl	x0, x0, #2
     9ac:	f94017e1 	ldr	x1, [sp, #40]
     9b0:	8b000020 	add	x0, x1, x0
     9b4:	bd400002 	ldr	s2, [x0]
     9b8:	b9403be1 	ldr	w1, [sp, #56]
     9bc:	b9400fe0 	ldr	w0, [sp, #12]
     9c0:	1b007c21 	mul	w1, w1, w0
     9c4:	b9403fe0 	ldr	w0, [sp, #60]
     9c8:	0b000020 	add	w0, w1, w0
     9cc:	2a0003e0 	mov	w0, w0
     9d0:	d37ef400 	lsl	x0, x0, #2
     9d4:	f94013e1 	ldr	x1, [sp, #32]
     9d8:	8b000020 	add	x0, x1, x0
     9dc:	bd400000 	ldr	s0, [x0]
     9e0:	1e200840 	fmul	s0, s2, s0
     9e4:	b9403be1 	ldr	w1, [sp, #56]
     9e8:	b94017e0 	ldr	w0, [sp, #20]
     9ec:	1b007c21 	mul	w1, w1, w0
     9f0:	b94037e0 	ldr	w0, [sp, #52]
     9f4:	0b000020 	add	w0, w1, w0
     9f8:	2a0003e0 	mov	w0, w0
     9fc:	d37ef400 	lsl	x0, x0, #2
     a00:	f9400fe1 	ldr	x1, [sp, #24]
     a04:	8b000020 	add	x0, x1, x0
     a08:	1e202820 	fadd	s0, s1, s0
     a0c:	bd000000 	str	s0, [x0]
     a10:	b9403fe0 	ldr	w0, [sp, #60]
     a14:	11000400 	add	w0, w0, #0x1
     a18:	b9003fe0 	str	w0, [sp, #60]
     a1c:	b9403fe0 	ldr	w0, [sp, #60]
     a20:	b9400fe1 	ldr	w1, [sp, #12]
     a24:	6b00003f 	cmp	w1, w0
     a28:	54fffa08 	b.hi	968 <matrix_multiply_c+0x5c>  // b.pmore
     a2c:	b9403be0 	ldr	w0, [sp, #56]
     a30:	11000400 	add	w0, w0, #0x1
     a34:	b9003be0 	str	w0, [sp, #56]
     a38:	b9403be0 	ldr	w0, [sp, #56]
     a3c:	b94013e1 	ldr	w1, [sp, #16]
     a40:	6b00003f 	cmp	w1, w0
     a44:	54fff7a8 	b.hi	938 <matrix_multiply_c+0x2c>  // b.pmore
     a48:	b94037e0 	ldr	w0, [sp, #52]
     a4c:	11000400 	add	w0, w0, #0x1
     a50:	b90037e0 	str	w0, [sp, #52]
     a54:	b94037e0 	ldr	w0, [sp, #52]
     a58:	b94017e1 	ldr	w1, [sp, #20]
     a5c:	6b00003f 	cmp	w1, w0
     a60:	54fff688 	b.hi	930 <matrix_multiply_c+0x24>  // b.pmore
     a64:	d503201f 	nop
     a68:	d503201f 	nop
     a6c:	910103ff 	add	sp, sp, #0x40
     a70:	d65f03c0 	ret

0000000000000a74 <matrix_multiply_neon>:
     a74:	d11103ff 	sub	sp, sp, #0x440
     a78:	a9007bfd 	stp	x29, x30, [sp]
     a7c:	910003fd 	mov	x29, sp
     a80:	f9001fe0 	str	x0, [sp, #56]
     a84:	f9001be1 	str	x1, [sp, #48]
     a88:	f90017e2 	str	x2, [sp, #40]
     a8c:	b90027e3 	str	w3, [sp, #36]
     a90:	b90023e4 	str	w4, [sp, #32]
     a94:	b9001fe5 	str	w5, [sp, #28]
     a98:	b0000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
     a9c:	f947f000 	ldr	x0, [x0, #4064]
     aa0:	f9400001 	ldr	x1, [x0]
     aa4:	f9021fe1 	str	x1, [sp, #1080]
     aa8:	d2800001 	mov	x1, #0x0                   	// #0
     aac:	b9004bff 	str	wzr, [sp, #72]
     ab0:	140001c4 	b	11c0 <matrix_multiply_neon+0x74c>
     ab4:	b9004fff 	str	wzr, [sp, #76]
     ab8:	140001bb 	b	11a4 <matrix_multiply_neon+0x730>
     abc:	b9007bff 	str	wzr, [sp, #120]
     ac0:	bd407be0 	ldr	s0, [sp, #120]
     ac4:	bd007fe0 	str	s0, [sp, #124]
     ac8:	bd407fe0 	ldr	s0, [sp, #124]
     acc:	4e040400 	dup	v0.4s, v0.s[0]
     ad0:	d503201f 	nop
     ad4:	3d804fe0 	str	q0, [sp, #304]
     ad8:	b90073ff 	str	wzr, [sp, #112]
     adc:	bd4073e0 	ldr	s0, [sp, #112]
     ae0:	bd0077e0 	str	s0, [sp, #116]
     ae4:	bd4077e0 	ldr	s0, [sp, #116]
     ae8:	4e040400 	dup	v0.4s, v0.s[0]
     aec:	d503201f 	nop
     af0:	3d8053e0 	str	q0, [sp, #320]
     af4:	b9006bff 	str	wzr, [sp, #104]
     af8:	bd406be0 	ldr	s0, [sp, #104]
     afc:	bd006fe0 	str	s0, [sp, #108]
     b00:	bd406fe0 	ldr	s0, [sp, #108]
     b04:	4e040400 	dup	v0.4s, v0.s[0]
     b08:	d503201f 	nop
     b0c:	3d8057e0 	str	q0, [sp, #336]
     b10:	b90063ff 	str	wzr, [sp, #96]
     b14:	bd4063e0 	ldr	s0, [sp, #96]
     b18:	bd0067e0 	str	s0, [sp, #100]
     b1c:	bd4067e0 	ldr	s0, [sp, #100]
     b20:	4e040400 	dup	v0.4s, v0.s[0]
     b24:	d503201f 	nop
     b28:	3d805be0 	str	q0, [sp, #352]
     b2c:	b90053ff 	str	wzr, [sp, #80]
     b30:	14000158 	b	1090 <matrix_multiply_neon+0x61c>
     b34:	b94053e1 	ldr	w1, [sp, #80]
     b38:	b94027e0 	ldr	w0, [sp, #36]
     b3c:	1b007c21 	mul	w1, w1, w0
     b40:	b9404be0 	ldr	w0, [sp, #72]
     b44:	0b000020 	add	w0, w1, w0
     b48:	b9005be0 	str	w0, [sp, #88]
     b4c:	b9404fe1 	ldr	w1, [sp, #76]
     b50:	b9401fe0 	ldr	w0, [sp, #28]
     b54:	1b007c21 	mul	w1, w1, w0
     b58:	b94053e0 	ldr	w0, [sp, #80]
     b5c:	0b000020 	add	w0, w1, w0
     b60:	b9005fe0 	str	w0, [sp, #92]
     b64:	b9805be0 	ldrsw	x0, [sp, #88]
     b68:	d37ef400 	lsl	x0, x0, #2
     b6c:	f9401fe1 	ldr	x1, [sp, #56]
     b70:	8b000020 	add	x0, x1, x0
     b74:	f9007fe0 	str	x0, [sp, #248]
     b78:	f9407fe0 	ldr	x0, [sp, #248]
     b7c:	3dc00000 	ldr	q0, [x0]
     b80:	3d805fe0 	str	q0, [sp, #368]
     b84:	b9805be1 	ldrsw	x1, [sp, #88]
     b88:	b94027e0 	ldr	w0, [sp, #36]
     b8c:	8b000020 	add	x0, x1, x0
     b90:	d37ef400 	lsl	x0, x0, #2
     b94:	f9401fe1 	ldr	x1, [sp, #56]
     b98:	8b000020 	add	x0, x1, x0
     b9c:	f9007be0 	str	x0, [sp, #240]
     ba0:	f9407be0 	ldr	x0, [sp, #240]
     ba4:	3dc00000 	ldr	q0, [x0]
     ba8:	3d8063e0 	str	q0, [sp, #384]
     bac:	b9805be1 	ldrsw	x1, [sp, #88]
     bb0:	b94027e0 	ldr	w0, [sp, #36]
     bb4:	531f7800 	lsl	w0, w0, #1
     bb8:	2a0003e0 	mov	w0, w0
     bbc:	8b000020 	add	x0, x1, x0
     bc0:	d37ef400 	lsl	x0, x0, #2
     bc4:	f9401fe1 	ldr	x1, [sp, #56]
     bc8:	8b000020 	add	x0, x1, x0
     bcc:	f90077e0 	str	x0, [sp, #232]
     bd0:	f94077e0 	ldr	x0, [sp, #232]
     bd4:	3dc00000 	ldr	q0, [x0]
     bd8:	3d8067e0 	str	q0, [sp, #400]
     bdc:	b9805be2 	ldrsw	x2, [sp, #88]
     be0:	b94027e1 	ldr	w1, [sp, #36]
     be4:	2a0103e0 	mov	w0, w1
     be8:	531f7800 	lsl	w0, w0, #1
     bec:	0b010000 	add	w0, w0, w1
     bf0:	2a0003e0 	mov	w0, w0
     bf4:	8b000040 	add	x0, x2, x0
     bf8:	d37ef400 	lsl	x0, x0, #2
     bfc:	f9401fe1 	ldr	x1, [sp, #56]
     c00:	8b000020 	add	x0, x1, x0
     c04:	f90073e0 	str	x0, [sp, #224]
     c08:	f94073e0 	ldr	x0, [sp, #224]
     c0c:	3dc00000 	ldr	q0, [x0]
     c10:	3d806be0 	str	q0, [sp, #416]
     c14:	b9805fe0 	ldrsw	x0, [sp, #92]
     c18:	d37ef400 	lsl	x0, x0, #2
     c1c:	f9401be1 	ldr	x1, [sp, #48]
     c20:	8b000020 	add	x0, x1, x0
     c24:	f9006fe0 	str	x0, [sp, #216]
     c28:	f9406fe0 	ldr	x0, [sp, #216]
     c2c:	3dc00000 	ldr	q0, [x0]
     c30:	3d806fe0 	str	q0, [sp, #432]
     c34:	3dc04fe0 	ldr	q0, [sp, #304]
     c38:	3d80f7e0 	str	q0, [sp, #976]
     c3c:	3dc05fe0 	ldr	q0, [sp, #368]
     c40:	3d80fbe0 	str	q0, [sp, #992]
     c44:	3dc06fe0 	ldr	q0, [sp, #432]
     c48:	3d804be0 	str	q0, [sp, #288]
     c4c:	bd4123e0 	ldr	s0, [sp, #288]
     c50:	bd00bfe0 	str	s0, [sp, #188]
     c54:	bd40bfe0 	ldr	s0, [sp, #188]
     c58:	4e040400 	dup	v0.4s, v0.s[0]
     c5c:	4ea01c02 	mov	v2.16b, v0.16b
     c60:	3dc0fbe1 	ldr	q1, [sp, #992]
     c64:	3dc0f7e0 	ldr	q0, [sp, #976]
     c68:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     c6c:	3d804fe0 	str	q0, [sp, #304]
     c70:	3dc04fe0 	ldr	q0, [sp, #304]
     c74:	3d80efe0 	str	q0, [sp, #944]
     c78:	3dc063e0 	ldr	q0, [sp, #384]
     c7c:	3d80f3e0 	str	q0, [sp, #960]
     c80:	3dc06fe0 	ldr	q0, [sp, #432]
     c84:	3d804be0 	str	q0, [sp, #288]
     c88:	bd4127e0 	ldr	s0, [sp, #292]
     c8c:	bd00bbe0 	str	s0, [sp, #184]
     c90:	bd40bbe0 	ldr	s0, [sp, #184]
     c94:	4e040400 	dup	v0.4s, v0.s[0]
     c98:	4ea01c02 	mov	v2.16b, v0.16b
     c9c:	3dc0f3e1 	ldr	q1, [sp, #960]
     ca0:	3dc0efe0 	ldr	q0, [sp, #944]
     ca4:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     ca8:	3d804fe0 	str	q0, [sp, #304]
     cac:	3dc04fe0 	ldr	q0, [sp, #304]
     cb0:	3d80e7e0 	str	q0, [sp, #912]
     cb4:	3dc067e0 	ldr	q0, [sp, #400]
     cb8:	3d80ebe0 	str	q0, [sp, #928]
     cbc:	3dc06fe0 	ldr	q0, [sp, #432]
     cc0:	3d804be0 	str	q0, [sp, #288]
     cc4:	bd412be0 	ldr	s0, [sp, #296]
     cc8:	bd00b7e0 	str	s0, [sp, #180]
     ccc:	bd40b7e0 	ldr	s0, [sp, #180]
     cd0:	4e040400 	dup	v0.4s, v0.s[0]
     cd4:	4ea01c02 	mov	v2.16b, v0.16b
     cd8:	3dc0ebe1 	ldr	q1, [sp, #928]
     cdc:	3dc0e7e0 	ldr	q0, [sp, #912]
     ce0:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     ce4:	3d804fe0 	str	q0, [sp, #304]
     ce8:	3dc04fe0 	ldr	q0, [sp, #304]
     cec:	3d80dfe0 	str	q0, [sp, #880]
     cf0:	3dc06be0 	ldr	q0, [sp, #416]
     cf4:	3d80e3e0 	str	q0, [sp, #896]
     cf8:	3dc06fe0 	ldr	q0, [sp, #432]
     cfc:	3d804be0 	str	q0, [sp, #288]
     d00:	bd412fe0 	ldr	s0, [sp, #300]
     d04:	bd00b3e0 	str	s0, [sp, #176]
     d08:	bd40b3e0 	ldr	s0, [sp, #176]
     d0c:	4e040400 	dup	v0.4s, v0.s[0]
     d10:	4ea01c02 	mov	v2.16b, v0.16b
     d14:	3dc0e3e1 	ldr	q1, [sp, #896]
     d18:	3dc0dfe0 	ldr	q0, [sp, #880]
     d1c:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     d20:	3d804fe0 	str	q0, [sp, #304]
     d24:	b9805fe1 	ldrsw	x1, [sp, #92]
     d28:	b9401fe0 	ldr	w0, [sp, #28]
     d2c:	8b000020 	add	x0, x1, x0
     d30:	d37ef400 	lsl	x0, x0, #2
     d34:	f9401be1 	ldr	x1, [sp, #48]
     d38:	8b000020 	add	x0, x1, x0
     d3c:	f9006be0 	str	x0, [sp, #208]
     d40:	f9406be0 	ldr	x0, [sp, #208]
     d44:	3dc00000 	ldr	q0, [x0]
     d48:	3d8073e0 	str	q0, [sp, #448]
     d4c:	3dc053e0 	ldr	q0, [sp, #320]
     d50:	3d80d7e0 	str	q0, [sp, #848]
     d54:	3dc05fe0 	ldr	q0, [sp, #368]
     d58:	3d80dbe0 	str	q0, [sp, #864]
     d5c:	3dc073e0 	ldr	q0, [sp, #448]
     d60:	3d804be0 	str	q0, [sp, #288]
     d64:	bd4123e0 	ldr	s0, [sp, #288]
     d68:	bd00afe0 	str	s0, [sp, #172]
     d6c:	bd40afe0 	ldr	s0, [sp, #172]
     d70:	4e040400 	dup	v0.4s, v0.s[0]
     d74:	4ea01c02 	mov	v2.16b, v0.16b
     d78:	3dc0dbe1 	ldr	q1, [sp, #864]
     d7c:	3dc0d7e0 	ldr	q0, [sp, #848]
     d80:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     d84:	3d8053e0 	str	q0, [sp, #320]
     d88:	3dc053e0 	ldr	q0, [sp, #320]
     d8c:	3d80cfe0 	str	q0, [sp, #816]
     d90:	3dc063e0 	ldr	q0, [sp, #384]
     d94:	3d80d3e0 	str	q0, [sp, #832]
     d98:	3dc073e0 	ldr	q0, [sp, #448]
     d9c:	3d804be0 	str	q0, [sp, #288]
     da0:	bd4127e0 	ldr	s0, [sp, #292]
     da4:	bd00abe0 	str	s0, [sp, #168]
     da8:	bd40abe0 	ldr	s0, [sp, #168]
     dac:	4e040400 	dup	v0.4s, v0.s[0]
     db0:	4ea01c02 	mov	v2.16b, v0.16b
     db4:	3dc0d3e1 	ldr	q1, [sp, #832]
     db8:	3dc0cfe0 	ldr	q0, [sp, #816]
     dbc:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     dc0:	3d8053e0 	str	q0, [sp, #320]
     dc4:	3dc053e0 	ldr	q0, [sp, #320]
     dc8:	3d80c7e0 	str	q0, [sp, #784]
     dcc:	3dc067e0 	ldr	q0, [sp, #400]
     dd0:	3d80cbe0 	str	q0, [sp, #800]
     dd4:	3dc073e0 	ldr	q0, [sp, #448]
     dd8:	3d804be0 	str	q0, [sp, #288]
     ddc:	bd412be0 	ldr	s0, [sp, #296]
     de0:	bd00a7e0 	str	s0, [sp, #164]
     de4:	bd40a7e0 	ldr	s0, [sp, #164]
     de8:	4e040400 	dup	v0.4s, v0.s[0]
     dec:	4ea01c02 	mov	v2.16b, v0.16b
     df0:	3dc0cbe1 	ldr	q1, [sp, #800]
     df4:	3dc0c7e0 	ldr	q0, [sp, #784]
     df8:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     dfc:	3d8053e0 	str	q0, [sp, #320]
     e00:	3dc053e0 	ldr	q0, [sp, #320]
     e04:	3d80bfe0 	str	q0, [sp, #752]
     e08:	3dc06be0 	ldr	q0, [sp, #416]
     e0c:	3d80c3e0 	str	q0, [sp, #768]
     e10:	3dc073e0 	ldr	q0, [sp, #448]
     e14:	3d804be0 	str	q0, [sp, #288]
     e18:	bd412fe0 	ldr	s0, [sp, #300]
     e1c:	bd00a3e0 	str	s0, [sp, #160]
     e20:	bd40a3e0 	ldr	s0, [sp, #160]
     e24:	4e040400 	dup	v0.4s, v0.s[0]
     e28:	4ea01c02 	mov	v2.16b, v0.16b
     e2c:	3dc0c3e1 	ldr	q1, [sp, #768]
     e30:	3dc0bfe0 	ldr	q0, [sp, #752]
     e34:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     e38:	3d8053e0 	str	q0, [sp, #320]
     e3c:	b9805fe1 	ldrsw	x1, [sp, #92]
     e40:	b9401fe0 	ldr	w0, [sp, #28]
     e44:	531f7800 	lsl	w0, w0, #1
     e48:	2a0003e0 	mov	w0, w0
     e4c:	8b000020 	add	x0, x1, x0
     e50:	d37ef400 	lsl	x0, x0, #2
     e54:	f9401be1 	ldr	x1, [sp, #48]
     e58:	8b000020 	add	x0, x1, x0
     e5c:	f90067e0 	str	x0, [sp, #200]
     e60:	f94067e0 	ldr	x0, [sp, #200]
     e64:	3dc00000 	ldr	q0, [x0]
     e68:	3d8077e0 	str	q0, [sp, #464]
     e6c:	3dc057e0 	ldr	q0, [sp, #336]
     e70:	3d80b7e0 	str	q0, [sp, #720]
     e74:	3dc05fe0 	ldr	q0, [sp, #368]
     e78:	3d80bbe0 	str	q0, [sp, #736]
     e7c:	3dc077e0 	ldr	q0, [sp, #464]
     e80:	3d804be0 	str	q0, [sp, #288]
     e84:	bd4123e0 	ldr	s0, [sp, #288]
     e88:	bd009fe0 	str	s0, [sp, #156]
     e8c:	bd409fe0 	ldr	s0, [sp, #156]
     e90:	4e040400 	dup	v0.4s, v0.s[0]
     e94:	4ea01c02 	mov	v2.16b, v0.16b
     e98:	3dc0bbe1 	ldr	q1, [sp, #736]
     e9c:	3dc0b7e0 	ldr	q0, [sp, #720]
     ea0:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     ea4:	3d8057e0 	str	q0, [sp, #336]
     ea8:	3dc057e0 	ldr	q0, [sp, #336]
     eac:	3d80afe0 	str	q0, [sp, #688]
     eb0:	3dc063e0 	ldr	q0, [sp, #384]
     eb4:	3d80b3e0 	str	q0, [sp, #704]
     eb8:	3dc077e0 	ldr	q0, [sp, #464]
     ebc:	3d804be0 	str	q0, [sp, #288]
     ec0:	bd4127e0 	ldr	s0, [sp, #292]
     ec4:	bd009be0 	str	s0, [sp, #152]
     ec8:	bd409be0 	ldr	s0, [sp, #152]
     ecc:	4e040400 	dup	v0.4s, v0.s[0]
     ed0:	4ea01c02 	mov	v2.16b, v0.16b
     ed4:	3dc0b3e1 	ldr	q1, [sp, #704]
     ed8:	3dc0afe0 	ldr	q0, [sp, #688]
     edc:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     ee0:	3d8057e0 	str	q0, [sp, #336]
     ee4:	3dc057e0 	ldr	q0, [sp, #336]
     ee8:	3d80a7e0 	str	q0, [sp, #656]
     eec:	3dc067e0 	ldr	q0, [sp, #400]
     ef0:	3d80abe0 	str	q0, [sp, #672]
     ef4:	3dc077e0 	ldr	q0, [sp, #464]
     ef8:	3d804be0 	str	q0, [sp, #288]
     efc:	bd412be0 	ldr	s0, [sp, #296]
     f00:	bd0097e0 	str	s0, [sp, #148]
     f04:	bd4097e0 	ldr	s0, [sp, #148]
     f08:	4e040400 	dup	v0.4s, v0.s[0]
     f0c:	4ea01c02 	mov	v2.16b, v0.16b
     f10:	3dc0abe1 	ldr	q1, [sp, #672]
     f14:	3dc0a7e0 	ldr	q0, [sp, #656]
     f18:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     f1c:	3d8057e0 	str	q0, [sp, #336]
     f20:	3dc057e0 	ldr	q0, [sp, #336]
     f24:	3d809fe0 	str	q0, [sp, #624]
     f28:	3dc06be0 	ldr	q0, [sp, #416]
     f2c:	3d80a3e0 	str	q0, [sp, #640]
     f30:	3dc077e0 	ldr	q0, [sp, #464]
     f34:	3d804be0 	str	q0, [sp, #288]
     f38:	bd412fe0 	ldr	s0, [sp, #300]
     f3c:	bd0093e0 	str	s0, [sp, #144]
     f40:	bd4093e0 	ldr	s0, [sp, #144]
     f44:	4e040400 	dup	v0.4s, v0.s[0]
     f48:	4ea01c02 	mov	v2.16b, v0.16b
     f4c:	3dc0a3e1 	ldr	q1, [sp, #640]
     f50:	3dc09fe0 	ldr	q0, [sp, #624]
     f54:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     f58:	3d8057e0 	str	q0, [sp, #336]
     f5c:	b9805fe2 	ldrsw	x2, [sp, #92]
     f60:	b9401fe1 	ldr	w1, [sp, #28]
     f64:	2a0103e0 	mov	w0, w1
     f68:	531f7800 	lsl	w0, w0, #1
     f6c:	0b010000 	add	w0, w0, w1
     f70:	2a0003e0 	mov	w0, w0
     f74:	8b000040 	add	x0, x2, x0
     f78:	d37ef400 	lsl	x0, x0, #2
     f7c:	f9401be1 	ldr	x1, [sp, #48]
     f80:	8b000020 	add	x0, x1, x0
     f84:	f90063e0 	str	x0, [sp, #192]
     f88:	f94063e0 	ldr	x0, [sp, #192]
     f8c:	3dc00000 	ldr	q0, [x0]
     f90:	3d807be0 	str	q0, [sp, #480]
     f94:	3dc05be0 	ldr	q0, [sp, #352]
     f98:	3d8097e0 	str	q0, [sp, #592]
     f9c:	3dc05fe0 	ldr	q0, [sp, #368]
     fa0:	3d809be0 	str	q0, [sp, #608]
     fa4:	3dc07be0 	ldr	q0, [sp, #480]
     fa8:	3d804be0 	str	q0, [sp, #288]
     fac:	bd4123e0 	ldr	s0, [sp, #288]
     fb0:	bd008fe0 	str	s0, [sp, #140]
     fb4:	bd408fe0 	ldr	s0, [sp, #140]
     fb8:	4e040400 	dup	v0.4s, v0.s[0]
     fbc:	4ea01c02 	mov	v2.16b, v0.16b
     fc0:	3dc09be1 	ldr	q1, [sp, #608]
     fc4:	3dc097e0 	ldr	q0, [sp, #592]
     fc8:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
     fcc:	3d805be0 	str	q0, [sp, #352]
     fd0:	3dc05be0 	ldr	q0, [sp, #352]
     fd4:	3d808fe0 	str	q0, [sp, #560]
     fd8:	3dc063e0 	ldr	q0, [sp, #384]
     fdc:	3d8093e0 	str	q0, [sp, #576]
     fe0:	3dc07be0 	ldr	q0, [sp, #480]
     fe4:	3d804be0 	str	q0, [sp, #288]
     fe8:	bd4127e0 	ldr	s0, [sp, #292]
     fec:	bd008be0 	str	s0, [sp, #136]
     ff0:	bd408be0 	ldr	s0, [sp, #136]
     ff4:	4e040400 	dup	v0.4s, v0.s[0]
     ff8:	4ea01c02 	mov	v2.16b, v0.16b
     ffc:	3dc093e1 	ldr	q1, [sp, #576]
    1000:	3dc08fe0 	ldr	q0, [sp, #560]
    1004:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
    1008:	3d805be0 	str	q0, [sp, #352]
    100c:	3dc05be0 	ldr	q0, [sp, #352]
    1010:	3d8087e0 	str	q0, [sp, #528]
    1014:	3dc067e0 	ldr	q0, [sp, #400]
    1018:	3d808be0 	str	q0, [sp, #544]
    101c:	3dc07be0 	ldr	q0, [sp, #480]
    1020:	3d804be0 	str	q0, [sp, #288]
    1024:	bd412be0 	ldr	s0, [sp, #296]
    1028:	bd0087e0 	str	s0, [sp, #132]
    102c:	bd4087e0 	ldr	s0, [sp, #132]
    1030:	4e040400 	dup	v0.4s, v0.s[0]
    1034:	4ea01c02 	mov	v2.16b, v0.16b
    1038:	3dc08be1 	ldr	q1, [sp, #544]
    103c:	3dc087e0 	ldr	q0, [sp, #528]
    1040:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
    1044:	3d805be0 	str	q0, [sp, #352]
    1048:	3dc05be0 	ldr	q0, [sp, #352]
    104c:	3d807fe0 	str	q0, [sp, #496]
    1050:	3dc06be0 	ldr	q0, [sp, #416]
    1054:	3d8083e0 	str	q0, [sp, #512]
    1058:	3dc07be0 	ldr	q0, [sp, #480]
    105c:	3d804be0 	str	q0, [sp, #288]
    1060:	bd412fe0 	ldr	s0, [sp, #300]
    1064:	bd0083e0 	str	s0, [sp, #128]
    1068:	bd4083e0 	ldr	s0, [sp, #128]
    106c:	4e040400 	dup	v0.4s, v0.s[0]
    1070:	4ea01c02 	mov	v2.16b, v0.16b
    1074:	3dc083e1 	ldr	q1, [sp, #512]
    1078:	3dc07fe0 	ldr	q0, [sp, #496]
    107c:	4e22cc20 	fmla	v0.4s, v1.4s, v2.4s
    1080:	3d805be0 	str	q0, [sp, #352]
    1084:	b94053e0 	ldr	w0, [sp, #80]
    1088:	11001000 	add	w0, w0, #0x4
    108c:	b90053e0 	str	w0, [sp, #80]
    1090:	b94053e0 	ldr	w0, [sp, #80]
    1094:	b9401fe1 	ldr	w1, [sp, #28]
    1098:	6b00003f 	cmp	w1, w0
    109c:	54ffd4c8 	b.hi	b34 <matrix_multiply_neon+0xc0>  // b.pmore
    10a0:	b9404fe1 	ldr	w1, [sp, #76]
    10a4:	b94027e0 	ldr	w0, [sp, #36]
    10a8:	1b007c21 	mul	w1, w1, w0
    10ac:	b9404be0 	ldr	w0, [sp, #72]
    10b0:	0b000020 	add	w0, w1, w0
    10b4:	b90057e0 	str	w0, [sp, #84]
    10b8:	b98057e0 	ldrsw	x0, [sp, #84]
    10bc:	d37ef400 	lsl	x0, x0, #2
    10c0:	f94017e1 	ldr	x1, [sp, #40]
    10c4:	8b000020 	add	x0, x1, x0
    10c8:	f9008fe0 	str	x0, [sp, #280]
    10cc:	3dc04fe0 	ldr	q0, [sp, #304]
    10d0:	3d810be0 	str	q0, [sp, #1056]
    10d4:	f9408fe0 	ldr	x0, [sp, #280]
    10d8:	3dc10be0 	ldr	q0, [sp, #1056]
    10dc:	3d800000 	str	q0, [x0]
    10e0:	d503201f 	nop
    10e4:	b98057e1 	ldrsw	x1, [sp, #84]
    10e8:	b94027e0 	ldr	w0, [sp, #36]
    10ec:	8b000020 	add	x0, x1, x0
    10f0:	d37ef400 	lsl	x0, x0, #2
    10f4:	f94017e1 	ldr	x1, [sp, #40]
    10f8:	8b000020 	add	x0, x1, x0
    10fc:	f9008be0 	str	x0, [sp, #272]
    1100:	3dc053e0 	ldr	q0, [sp, #320]
    1104:	3d8107e0 	str	q0, [sp, #1040]
    1108:	f9408be0 	ldr	x0, [sp, #272]
    110c:	3dc107e0 	ldr	q0, [sp, #1040]
    1110:	3d800000 	str	q0, [x0]
    1114:	d503201f 	nop
    1118:	b98057e1 	ldrsw	x1, [sp, #84]
    111c:	b94027e0 	ldr	w0, [sp, #36]
    1120:	531f7800 	lsl	w0, w0, #1
    1124:	2a0003e0 	mov	w0, w0
    1128:	8b000020 	add	x0, x1, x0
    112c:	d37ef400 	lsl	x0, x0, #2
    1130:	f94017e1 	ldr	x1, [sp, #40]
    1134:	8b000020 	add	x0, x1, x0
    1138:	f90087e0 	str	x0, [sp, #264]
    113c:	3dc057e0 	ldr	q0, [sp, #336]
    1140:	3d8103e0 	str	q0, [sp, #1024]
    1144:	f94087e0 	ldr	x0, [sp, #264]
    1148:	3dc103e0 	ldr	q0, [sp, #1024]
    114c:	3d800000 	str	q0, [x0]
    1150:	d503201f 	nop
    1154:	b98057e2 	ldrsw	x2, [sp, #84]
    1158:	b94027e1 	ldr	w1, [sp, #36]
    115c:	2a0103e0 	mov	w0, w1
    1160:	531f7800 	lsl	w0, w0, #1
    1164:	0b010000 	add	w0, w0, w1
    1168:	2a0003e0 	mov	w0, w0
    116c:	8b000040 	add	x0, x2, x0
    1170:	d37ef400 	lsl	x0, x0, #2
    1174:	f94017e1 	ldr	x1, [sp, #40]
    1178:	8b000020 	add	x0, x1, x0
    117c:	f90083e0 	str	x0, [sp, #256]
    1180:	3dc05be0 	ldr	q0, [sp, #352]
    1184:	3d80ffe0 	str	q0, [sp, #1008]
    1188:	f94083e0 	ldr	x0, [sp, #256]
    118c:	3dc0ffe0 	ldr	q0, [sp, #1008]
    1190:	3d800000 	str	q0, [x0]
    1194:	d503201f 	nop
    1198:	b9404fe0 	ldr	w0, [sp, #76]
    119c:	11001000 	add	w0, w0, #0x4
    11a0:	b9004fe0 	str	w0, [sp, #76]
    11a4:	b9404fe0 	ldr	w0, [sp, #76]
    11a8:	b94023e1 	ldr	w1, [sp, #32]
    11ac:	6b00003f 	cmp	w1, w0
    11b0:	54ffc868 	b.hi	abc <matrix_multiply_neon+0x48>  // b.pmore
    11b4:	b9404be0 	ldr	w0, [sp, #72]
    11b8:	11001000 	add	w0, w0, #0x4
    11bc:	b9004be0 	str	w0, [sp, #72]
    11c0:	b9404be0 	ldr	w0, [sp, #72]
    11c4:	b94027e1 	ldr	w1, [sp, #36]
    11c8:	6b00003f 	cmp	w1, w0
    11cc:	54ffc748 	b.hi	ab4 <matrix_multiply_neon+0x40>  // b.pmore
    11d0:	d503201f 	nop
    11d4:	90000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
    11d8:	f947f000 	ldr	x0, [x0, #4064]
    11dc:	f9421fe1 	ldr	x1, [sp, #1080]
    11e0:	f9400002 	ldr	x2, [x0]
    11e4:	eb020021 	subs	x1, x1, x2
    11e8:	d2800002 	mov	x2, #0x0                   	// #0
    11ec:	54000040 	b.eq	11f4 <matrix_multiply_neon+0x780>  // b.none
    11f0:	97fffd6c 	bl	7a0 <__stack_chk_fail@plt>
    11f4:	a9407bfd 	ldp	x29, x30, [sp]
    11f8:	911103ff 	add	sp, sp, #0x440
    11fc:	d65f03c0 	ret

0000000000001200 <print_matrix>:
    1200:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1204:	910003fd 	mov	x29, sp
    1208:	f9000fe0 	str	x0, [sp, #24]
    120c:	b90017e1 	str	w1, [sp, #20]
    1210:	b90013e2 	str	w2, [sp, #16]
    1214:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    1218:	9127a000 	add	x0, x0, #0x9e8
    121c:	97fffd6d 	bl	7d0 <puts@plt>
    1220:	b9002bff 	str	wzr, [sp, #40]
    1224:	1400001d 	b	1298 <print_matrix+0x98>
    1228:	b9002fff 	str	wzr, [sp, #44]
    122c:	14000012 	b	1274 <print_matrix+0x74>
    1230:	b9402fe1 	ldr	w1, [sp, #44]
    1234:	b94017e0 	ldr	w0, [sp, #20]
    1238:	1b007c21 	mul	w1, w1, w0
    123c:	b9402be0 	ldr	w0, [sp, #40]
    1240:	0b000020 	add	w0, w1, w0
    1244:	2a0003e0 	mov	w0, w0
    1248:	d37ef400 	lsl	x0, x0, #2
    124c:	f9400fe1 	ldr	x1, [sp, #24]
    1250:	8b000020 	add	x0, x1, x0
    1254:	bd400000 	ldr	s0, [x0]
    1258:	1e22c000 	fcvt	d0, s0
    125c:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    1260:	91282000 	add	x0, x0, #0xa08
    1264:	97fffd5f 	bl	7e0 <printf@plt>
    1268:	b9402fe0 	ldr	w0, [sp, #44]
    126c:	11000400 	add	w0, w0, #0x1
    1270:	b9002fe0 	str	w0, [sp, #44]
    1274:	b9402fe0 	ldr	w0, [sp, #44]
    1278:	b94013e1 	ldr	w1, [sp, #16]
    127c:	6b00003f 	cmp	w1, w0
    1280:	54fffd88 	b.hi	1230 <print_matrix+0x30>  // b.pmore
    1284:	52800140 	mov	w0, #0xa                   	// #10
    1288:	97fffd5a 	bl	7f0 <putchar@plt>
    128c:	b9402be0 	ldr	w0, [sp, #40]
    1290:	11000400 	add	w0, w0, #0x1
    1294:	b9002be0 	str	w0, [sp, #40]
    1298:	b9402be0 	ldr	w0, [sp, #40]
    129c:	b94017e1 	ldr	w1, [sp, #20]
    12a0:	6b00003f 	cmp	w1, w0
    12a4:	54fffc28 	b.hi	1228 <print_matrix+0x28>  // b.pmore
    12a8:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    12ac:	91284000 	add	x0, x0, #0xa10
    12b0:	97fffd48 	bl	7d0 <puts@plt>
    12b4:	d503201f 	nop
    12b8:	a8c37bfd 	ldp	x29, x30, [sp], #48
    12bc:	d65f03c0 	ret

00000000000012c0 <init_matrix_rand>:
    12c0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    12c4:	910003fd 	mov	x29, sp
    12c8:	f9000fe0 	str	x0, [sp, #24]
    12cc:	b90017e1 	str	w1, [sp, #20]
    12d0:	b9002fff 	str	wzr, [sp, #44]
    12d4:	1400000d 	b	1308 <init_matrix_rand+0x48>
    12d8:	97fffd2e 	bl	790 <rand@plt>
    12dc:	1e220001 	scvtf	s1, w0
    12e0:	b9802fe0 	ldrsw	x0, [sp, #44]
    12e4:	d37ef400 	lsl	x0, x0, #2
    12e8:	f9400fe1 	ldr	x1, [sp, #24]
    12ec:	8b000020 	add	x0, x1, x0
    12f0:	0f0265e0 	movi	v0.2s, #0x4f, lsl #24
    12f4:	1e201820 	fdiv	s0, s1, s0
    12f8:	bd000000 	str	s0, [x0]
    12fc:	b9402fe0 	ldr	w0, [sp, #44]
    1300:	11000400 	add	w0, w0, #0x1
    1304:	b9002fe0 	str	w0, [sp, #44]
    1308:	b9402fe0 	ldr	w0, [sp, #44]
    130c:	b94017e1 	ldr	w1, [sp, #20]
    1310:	6b00003f 	cmp	w1, w0
    1314:	54fffe28 	b.hi	12d8 <init_matrix_rand+0x18>  // b.pmore
    1318:	d503201f 	nop
    131c:	d503201f 	nop
    1320:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1324:	d65f03c0 	ret

0000000000001328 <init_matrix_zero>:
    1328:	d10083ff 	sub	sp, sp, #0x20
    132c:	f90007e0 	str	x0, [sp, #8]
    1330:	b90007e1 	str	w1, [sp, #4]
    1334:	b9001fff 	str	wzr, [sp, #28]
    1338:	14000009 	b	135c <init_matrix_zero+0x34>
    133c:	b9801fe0 	ldrsw	x0, [sp, #28]
    1340:	d37ef400 	lsl	x0, x0, #2
    1344:	f94007e1 	ldr	x1, [sp, #8]
    1348:	8b000020 	add	x0, x1, x0
    134c:	b900001f 	str	wzr, [x0]
    1350:	b9401fe0 	ldr	w0, [sp, #28]
    1354:	11000400 	add	w0, w0, #0x1
    1358:	b9001fe0 	str	w0, [sp, #28]
    135c:	b9401fe0 	ldr	w0, [sp, #28]
    1360:	b94007e1 	ldr	w1, [sp, #4]
    1364:	6b00003f 	cmp	w1, w0
    1368:	54fffea8 	b.hi	133c <init_matrix_zero+0x14>  // b.pmore
    136c:	d503201f 	nop
    1370:	d503201f 	nop
    1374:	910083ff 	add	sp, sp, #0x20
    1378:	d65f03c0 	ret

000000000000137c <is_ele_equal>:
    137c:	d10043ff 	sub	sp, sp, #0x10
    1380:	bd000fe0 	str	s0, [sp, #12]
    1384:	bd000be1 	str	s1, [sp, #8]
    1388:	bd400fe1 	ldr	s1, [sp, #12]
    138c:	bd400be0 	ldr	s0, [sp, #8]
    1390:	1e203820 	fsub	s0, s1, s0
    1394:	1e20c000 	fabs	s0, s0
    1398:	1e22c000 	fcvt	d0, s0
    139c:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    13a0:	fd453401 	ldr	d1, [x0, #2664]
    13a4:	1e612010 	fcmpe	d0, d1
    13a8:	54000065 	b.pl	13b4 <is_ele_equal+0x38>  // b.nfrst
    13ac:	52800020 	mov	w0, #0x1                   	// #1
    13b0:	14000002 	b	13b8 <is_ele_equal+0x3c>
    13b4:	52800000 	mov	w0, #0x0                   	// #0
    13b8:	910043ff 	add	sp, sp, #0x10
    13bc:	d65f03c0 	ret

00000000000013c0 <compare_matrix>:
    13c0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    13c4:	910003fd 	mov	x29, sp
    13c8:	f90017e0 	str	x0, [sp, #40]
    13cc:	f90013e1 	str	x1, [sp, #32]
    13d0:	b9001fe2 	str	w2, [sp, #28]
    13d4:	b9001be3 	str	w3, [sp, #24]
    13d8:	b90033ff 	str	wzr, [sp, #48]
    13dc:	14000033 	b	14a8 <compare_matrix+0xe8>
    13e0:	b90037ff 	str	wzr, [sp, #52]
    13e4:	1400002a 	b	148c <compare_matrix+0xcc>
    13e8:	b94037e1 	ldr	w1, [sp, #52]
    13ec:	b9401fe0 	ldr	w0, [sp, #28]
    13f0:	1b007c21 	mul	w1, w1, w0
    13f4:	b94033e0 	ldr	w0, [sp, #48]
    13f8:	0b000020 	add	w0, w1, w0
    13fc:	2a0003e0 	mov	w0, w0
    1400:	d37ef400 	lsl	x0, x0, #2
    1404:	f94017e1 	ldr	x1, [sp, #40]
    1408:	8b000020 	add	x0, x1, x0
    140c:	bd400000 	ldr	s0, [x0]
    1410:	5ea1b800 	fcvtzs	s0, s0
    1414:	bd003be0 	str	s0, [sp, #56]
    1418:	b94037e1 	ldr	w1, [sp, #52]
    141c:	b9401fe0 	ldr	w0, [sp, #28]
    1420:	1b007c21 	mul	w1, w1, w0
    1424:	b94033e0 	ldr	w0, [sp, #48]
    1428:	0b000020 	add	w0, w1, w0
    142c:	2a0003e0 	mov	w0, w0
    1430:	d37ef400 	lsl	x0, x0, #2
    1434:	f94013e1 	ldr	x1, [sp, #32]
    1438:	8b000020 	add	x0, x1, x0
    143c:	bd400000 	ldr	s0, [x0]
    1440:	5ea1b800 	fcvtzs	s0, s0
    1444:	bd003fe0 	str	s0, [sp, #60]
    1448:	bd403be0 	ldr	s0, [sp, #56]
    144c:	5e21d802 	scvtf	s2, s0
    1450:	bd403fe0 	ldr	s0, [sp, #60]
    1454:	5e21d800 	scvtf	s0, s0
    1458:	1e204001 	fmov	s1, s0
    145c:	1e204040 	fmov	s0, s2
    1460:	97ffffc7 	bl	137c <is_ele_equal>
    1464:	12001c00 	and	w0, w0, #0xff
    1468:	52000000 	eor	w0, w0, #0x1
    146c:	12001c00 	and	w0, w0, #0xff
    1470:	7100001f 	cmp	w0, #0x0
    1474:	54000060 	b.eq	1480 <compare_matrix+0xc0>  // b.none
    1478:	52800000 	mov	w0, #0x0                   	// #0
    147c:	14000010 	b	14bc <compare_matrix+0xfc>
    1480:	b94037e0 	ldr	w0, [sp, #52]
    1484:	11000400 	add	w0, w0, #0x1
    1488:	b90037e0 	str	w0, [sp, #52]
    148c:	b94037e0 	ldr	w0, [sp, #52]
    1490:	b9401be1 	ldr	w1, [sp, #24]
    1494:	6b00003f 	cmp	w1, w0
    1498:	54fffa88 	b.hi	13e8 <compare_matrix+0x28>  // b.pmore
    149c:	b94033e0 	ldr	w0, [sp, #48]
    14a0:	11000400 	add	w0, w0, #0x1
    14a4:	b90033e0 	str	w0, [sp, #48]
    14a8:	b94033e0 	ldr	w0, [sp, #48]
    14ac:	b9401fe1 	ldr	w1, [sp, #28]
    14b0:	6b00003f 	cmp	w1, w0
    14b4:	54fff968 	b.hi	13e0 <compare_matrix+0x20>  // b.pmore
    14b8:	52800020 	mov	w0, #0x1                   	// #1
    14bc:	a8c47bfd 	ldp	x29, x30, [sp], #64
    14c0:	d65f03c0 	ret

00000000000014c4 <main>:
    14c4:	a9b17bfd 	stp	x29, x30, [sp, #-240]!
    14c8:	910003fd 	mov	x29, sp
    14cc:	a90153f3 	stp	x19, x20, [sp, #16]
    14d0:	a9025bf5 	stp	x21, x22, [sp, #32]
    14d4:	a90363f7 	stp	x23, x24, [sp, #48]
    14d8:	a9046bf9 	stp	x25, x26, [sp, #64]
    14dc:	a90573fb 	stp	x27, x28, [sp, #80]
    14e0:	d10043ff 	sub	sp, sp, #0x10
    14e4:	90000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
    14e8:	f947f000 	ldr	x0, [x0, #4064]
    14ec:	f9400001 	ldr	x1, [x0]
    14f0:	f90077a1 	str	x1, [x29, #232]
    14f4:	d2800001 	mov	x1, #0x0                   	// #0
    14f8:	910003e0 	mov	x0, sp
    14fc:	aa0003fc 	mov	x28, x0
    1500:	52800100 	mov	w0, #0x8                   	// #8
    1504:	b9009fa0 	str	w0, [x29, #156]
    1508:	52800100 	mov	w0, #0x8                   	// #8
    150c:	b9009ba0 	str	w0, [x29, #152]
    1510:	52800100 	mov	w0, #0x8                   	// #8
    1514:	b90097a0 	str	w0, [x29, #148]
    1518:	b9409fa1 	ldr	w1, [x29, #156]
    151c:	b94097a0 	ldr	w0, [x29, #148]
    1520:	1b007c21 	mul	w1, w1, w0
    1524:	2a0103e0 	mov	w0, w1
    1528:	d1000400 	sub	x0, x0, #0x1
    152c:	f9005ba0 	str	x0, [x29, #176]
    1530:	2a0103e0 	mov	w0, w1
    1534:	f90043a0 	str	x0, [x29, #128]
    1538:	f90047bf 	str	xzr, [x29, #136]
    153c:	f94043a0 	ldr	x0, [x29, #128]
    1540:	d37bfc00 	lsr	x0, x0, #59
    1544:	f94047be 	ldr	x30, [x29, #136]
    1548:	d37bebc3 	lsl	x3, x30, #5
    154c:	aa030003 	orr	x3, x0, x3
    1550:	f94043a0 	ldr	x0, [x29, #128]
    1554:	d37be802 	lsl	x2, x0, #5
    1558:	2a0103e0 	mov	w0, w1
    155c:	f9003ba0 	str	x0, [x29, #112]
    1560:	f9003fbf 	str	xzr, [x29, #120]
    1564:	a9470fa2 	ldp	x2, x3, [x29, #112]
    1568:	aa0203e0 	mov	x0, x2
    156c:	d37bfc00 	lsr	x0, x0, #59
    1570:	aa0303fe 	mov	x30, x3
    1574:	d37bebd1 	lsl	x17, x30, #5
    1578:	aa110011 	orr	x17, x0, x17
    157c:	aa0203e0 	mov	x0, x2
    1580:	d37be810 	lsl	x16, x0, #5
    1584:	2a0103e0 	mov	w0, w1
    1588:	d37ef400 	lsl	x0, x0, #2
    158c:	91003c00 	add	x0, x0, #0xf
    1590:	d344fc00 	lsr	x0, x0, #4
    1594:	d37cec00 	lsl	x0, x0, #4
    1598:	9270bc01 	and	x1, x0, #0xffffffffffff0000
    159c:	cb2163e1 	sub	x1, sp, x1
    15a0:	eb2163ff 	cmp	sp, x1
    15a4:	54000080 	b.eq	15b4 <main+0xf0>  // b.none
    15a8:	d14043ff 	sub	sp, sp, #0x10, lsl #12
    15ac:	f90203ff 	str	xzr, [sp, #1024]
    15b0:	17fffffc 	b	15a0 <main+0xdc>
    15b4:	92403c01 	and	x1, x0, #0xffff
    15b8:	cb2163ff 	sub	sp, sp, x1
    15bc:	f90003ff 	str	xzr, [sp]
    15c0:	92403c00 	and	x0, x0, #0xffff
    15c4:	f110001f 	cmp	x0, #0x400
    15c8:	54000043 	b.cc	15d0 <main+0x10c>  // b.lo, b.ul, b.last
    15cc:	f90203ff 	str	xzr, [sp, #1024]
    15d0:	910043e0 	add	x0, sp, #0x10
    15d4:	91000c00 	add	x0, x0, #0x3
    15d8:	d342fc00 	lsr	x0, x0, #2
    15dc:	d37ef400 	lsl	x0, x0, #2
    15e0:	f90057a0 	str	x0, [x29, #168]
    15e4:	b94097a1 	ldr	w1, [x29, #148]
    15e8:	b9409ba0 	ldr	w0, [x29, #152]
    15ec:	1b007c20 	mul	w0, w1, w0
    15f0:	2a0003e1 	mov	w1, w0
    15f4:	d1000421 	sub	x1, x1, #0x1
    15f8:	f90053a1 	str	x1, [x29, #160]
    15fc:	2a0003e1 	mov	w1, w0
    1600:	f90033a1 	str	x1, [x29, #96]
    1604:	f90037bf 	str	xzr, [x29, #104]
    1608:	a9460fa2 	ldp	x2, x3, [x29, #96]
    160c:	aa0203e1 	mov	x1, x2
    1610:	d37bfc21 	lsr	x1, x1, #59
    1614:	aa0303f0 	mov	x16, x3
    1618:	d37bea0f 	lsl	x15, x16, #5
    161c:	aa0f002f 	orr	x15, x1, x15
    1620:	aa0203e1 	mov	x1, x2
    1624:	d37be82e 	lsl	x14, x1, #5
    1628:	2a0003e1 	mov	w1, w0
    162c:	aa0103fa 	mov	x26, x1
    1630:	d280001b 	mov	x27, #0x0                   	// #0
    1634:	d37bff41 	lsr	x1, x26, #59
    1638:	d37beb6d 	lsl	x13, x27, #5
    163c:	aa0d002d 	orr	x13, x1, x13
    1640:	d37beb4c 	lsl	x12, x26, #5
    1644:	2a0003e0 	mov	w0, w0
    1648:	d37ef400 	lsl	x0, x0, #2
    164c:	91003c00 	add	x0, x0, #0xf
    1650:	d344fc00 	lsr	x0, x0, #4
    1654:	d37cec00 	lsl	x0, x0, #4
    1658:	9270bc01 	and	x1, x0, #0xffffffffffff0000
    165c:	cb2163e1 	sub	x1, sp, x1
    1660:	eb2163ff 	cmp	sp, x1
    1664:	54000080 	b.eq	1674 <main+0x1b0>  // b.none
    1668:	d14043ff 	sub	sp, sp, #0x10, lsl #12
    166c:	f90203ff 	str	xzr, [sp, #1024]
    1670:	17fffffc 	b	1660 <main+0x19c>
    1674:	92403c01 	and	x1, x0, #0xffff
    1678:	cb2163ff 	sub	sp, sp, x1
    167c:	f90003ff 	str	xzr, [sp]
    1680:	92403c00 	and	x0, x0, #0xffff
    1684:	f110001f 	cmp	x0, #0x400
    1688:	54000043 	b.cc	1690 <main+0x1cc>  // b.lo, b.ul, b.last
    168c:	f90203ff 	str	xzr, [sp, #1024]
    1690:	910043e0 	add	x0, sp, #0x10
    1694:	91000c00 	add	x0, x0, #0x3
    1698:	d342fc00 	lsr	x0, x0, #2
    169c:	d37ef400 	lsl	x0, x0, #2
    16a0:	f9005fa0 	str	x0, [x29, #184]
    16a4:	b9409fa1 	ldr	w1, [x29, #156]
    16a8:	b9409ba0 	ldr	w0, [x29, #152]
    16ac:	1b007c20 	mul	w0, w1, w0
    16b0:	2a0003e1 	mov	w1, w0
    16b4:	d1000421 	sub	x1, x1, #0x1
    16b8:	f90063a1 	str	x1, [x29, #192]
    16bc:	2a0003e1 	mov	w1, w0
    16c0:	aa0103f8 	mov	x24, x1
    16c4:	d2800019 	mov	x25, #0x0                   	// #0
    16c8:	d37bff01 	lsr	x1, x24, #59
    16cc:	d37beb2b 	lsl	x11, x25, #5
    16d0:	aa0b002b 	orr	x11, x1, x11
    16d4:	d37beb0a 	lsl	x10, x24, #5
    16d8:	2a0003e1 	mov	w1, w0
    16dc:	aa0103f6 	mov	x22, x1
    16e0:	d2800017 	mov	x23, #0x0                   	// #0
    16e4:	d37bfec1 	lsr	x1, x22, #59
    16e8:	d37beae9 	lsl	x9, x23, #5
    16ec:	aa090029 	orr	x9, x1, x9
    16f0:	d37beac8 	lsl	x8, x22, #5
    16f4:	2a0003e0 	mov	w0, w0
    16f8:	d37ef400 	lsl	x0, x0, #2
    16fc:	91003c00 	add	x0, x0, #0xf
    1700:	d344fc00 	lsr	x0, x0, #4
    1704:	d37cec00 	lsl	x0, x0, #4
    1708:	9270bc01 	and	x1, x0, #0xffffffffffff0000
    170c:	cb2163e1 	sub	x1, sp, x1
    1710:	eb2163ff 	cmp	sp, x1
    1714:	54000080 	b.eq	1724 <main+0x260>  // b.none
    1718:	d14043ff 	sub	sp, sp, #0x10, lsl #12
    171c:	f90203ff 	str	xzr, [sp, #1024]
    1720:	17fffffc 	b	1710 <main+0x24c>
    1724:	92403c01 	and	x1, x0, #0xffff
    1728:	cb2163ff 	sub	sp, sp, x1
    172c:	f90003ff 	str	xzr, [sp]
    1730:	92403c00 	and	x0, x0, #0xffff
    1734:	f110001f 	cmp	x0, #0x400
    1738:	54000043 	b.cc	1740 <main+0x27c>  // b.lo, b.ul, b.last
    173c:	f90203ff 	str	xzr, [sp, #1024]
    1740:	910043e0 	add	x0, sp, #0x10
    1744:	91000c00 	add	x0, x0, #0x3
    1748:	d342fc00 	lsr	x0, x0, #2
    174c:	d37ef400 	lsl	x0, x0, #2
    1750:	f90067a0 	str	x0, [x29, #200]
    1754:	b9409fa1 	ldr	w1, [x29, #156]
    1758:	b9409ba0 	ldr	w0, [x29, #152]
    175c:	1b007c20 	mul	w0, w1, w0
    1760:	2a0003e1 	mov	w1, w0
    1764:	d1000421 	sub	x1, x1, #0x1
    1768:	f9006ba1 	str	x1, [x29, #208]
    176c:	2a0003e1 	mov	w1, w0
    1770:	aa0103f4 	mov	x20, x1
    1774:	d2800015 	mov	x21, #0x0                   	// #0
    1778:	d37bfe81 	lsr	x1, x20, #59
    177c:	d37beaa7 	lsl	x7, x21, #5
    1780:	aa070027 	orr	x7, x1, x7
    1784:	d37bea86 	lsl	x6, x20, #5
    1788:	2a0003e1 	mov	w1, w0
    178c:	aa0103f2 	mov	x18, x1
    1790:	d2800013 	mov	x19, #0x0                   	// #0
    1794:	d37bfe41 	lsr	x1, x18, #59
    1798:	d37bea65 	lsl	x5, x19, #5
    179c:	aa050025 	orr	x5, x1, x5
    17a0:	d37bea44 	lsl	x4, x18, #5
    17a4:	2a0003e0 	mov	w0, w0
    17a8:	d37ef400 	lsl	x0, x0, #2
    17ac:	91003c00 	add	x0, x0, #0xf
    17b0:	d344fc00 	lsr	x0, x0, #4
    17b4:	d37cec00 	lsl	x0, x0, #4
    17b8:	9270bc01 	and	x1, x0, #0xffffffffffff0000
    17bc:	cb2163e1 	sub	x1, sp, x1
    17c0:	eb2163ff 	cmp	sp, x1
    17c4:	54000080 	b.eq	17d4 <main+0x310>  // b.none
    17c8:	d14043ff 	sub	sp, sp, #0x10, lsl #12
    17cc:	f90203ff 	str	xzr, [sp, #1024]
    17d0:	17fffffc 	b	17c0 <main+0x2fc>
    17d4:	92403c01 	and	x1, x0, #0xffff
    17d8:	cb2163ff 	sub	sp, sp, x1
    17dc:	f90003ff 	str	xzr, [sp]
    17e0:	92403c00 	and	x0, x0, #0xffff
    17e4:	f110001f 	cmp	x0, #0x400
    17e8:	54000043 	b.cc	17f0 <main+0x32c>  // b.lo, b.ul, b.last
    17ec:	f90203ff 	str	xzr, [sp, #1024]
    17f0:	910043e0 	add	x0, sp, #0x10
    17f4:	91000c00 	add	x0, x0, #0x3
    17f8:	d342fc00 	lsr	x0, x0, #2
    17fc:	d37ef400 	lsl	x0, x0, #2
    1800:	f9006fa0 	str	x0, [x29, #216]
    1804:	b9409fa1 	ldr	w1, [x29, #156]
    1808:	b94097a0 	ldr	w0, [x29, #148]
    180c:	1b007c20 	mul	w0, w1, w0
    1810:	2a0003e1 	mov	w1, w0
    1814:	f94057a0 	ldr	x0, [x29, #168]
    1818:	97fffeaa 	bl	12c0 <init_matrix_rand>
    181c:	b94097a2 	ldr	w2, [x29, #148]
    1820:	b9409fa1 	ldr	w1, [x29, #156]
    1824:	f94057a0 	ldr	x0, [x29, #168]
    1828:	97fffe76 	bl	1200 <print_matrix>
    182c:	b94097a1 	ldr	w1, [x29, #148]
    1830:	b9409ba0 	ldr	w0, [x29, #152]
    1834:	1b007c20 	mul	w0, w1, w0
    1838:	2a0003e1 	mov	w1, w0
    183c:	f9405fa0 	ldr	x0, [x29, #184]
    1840:	97fffea0 	bl	12c0 <init_matrix_rand>
    1844:	b9409ba2 	ldr	w2, [x29, #152]
    1848:	b94097a1 	ldr	w1, [x29, #148]
    184c:	f9405fa0 	ldr	x0, [x29, #184]
    1850:	97fffe6c 	bl	1200 <print_matrix>
    1854:	b94097a5 	ldr	w5, [x29, #148]
    1858:	b9409ba4 	ldr	w4, [x29, #152]
    185c:	b9409fa3 	ldr	w3, [x29, #156]
    1860:	f9406fa2 	ldr	x2, [x29, #216]
    1864:	f9405fa1 	ldr	x1, [x29, #184]
    1868:	f94057a0 	ldr	x0, [x29, #168]
    186c:	97fffc28 	bl	90c <matrix_multiply_c>
    1870:	b9409ba2 	ldr	w2, [x29, #152]
    1874:	b9409fa1 	ldr	w1, [x29, #156]
    1878:	f9406fa0 	ldr	x0, [x29, #216]
    187c:	97fffe61 	bl	1200 <print_matrix>
    1880:	b94097a5 	ldr	w5, [x29, #148]
    1884:	b9409ba4 	ldr	w4, [x29, #152]
    1888:	b9409fa3 	ldr	w3, [x29, #156]
    188c:	f94067a2 	ldr	x2, [x29, #200]
    1890:	f9405fa1 	ldr	x1, [x29, #184]
    1894:	f94057a0 	ldr	x0, [x29, #168]
    1898:	97fffc77 	bl	a74 <matrix_multiply_neon>
    189c:	b9409ba2 	ldr	w2, [x29, #152]
    18a0:	b9409fa1 	ldr	w1, [x29, #156]
    18a4:	f94067a0 	ldr	x0, [x29, #200]
    18a8:	97fffe56 	bl	1200 <print_matrix>
    18ac:	b9409ba3 	ldr	w3, [x29, #152]
    18b0:	b9409fa2 	ldr	w2, [x29, #156]
    18b4:	f9405fa1 	ldr	x1, [x29, #184]
    18b8:	f94057a0 	ldr	x0, [x29, #168]
    18bc:	97fffec1 	bl	13c0 <compare_matrix>
    18c0:	12001c00 	and	w0, w0, #0xff
    18c4:	7100001f 	cmp	w0, #0x0
    18c8:	54000080 	b.eq	18d8 <main+0x414>  // b.none
    18cc:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    18d0:	9128c000 	add	x0, x0, #0xa30
    18d4:	14000003 	b	18e0 <main+0x41c>
    18d8:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    18dc:	9128e000 	add	x0, x0, #0xa38
    18e0:	f90073a0 	str	x0, [x29, #224]
    18e4:	f94073a1 	ldr	x1, [x29, #224]
    18e8:	90000000 	adrp	x0, 1000 <matrix_multiply_neon+0x58c>
    18ec:	91292000 	add	x0, x0, #0xa48
    18f0:	97fffbbc 	bl	7e0 <printf@plt>
    18f4:	52800000 	mov	w0, #0x0                   	// #0
    18f8:	9100039f 	mov	sp, x28
    18fc:	2a0003e1 	mov	w1, w0
    1900:	90000080 	adrp	x0, 11000 <__FRAME_END__+0xf340>
    1904:	f947f000 	ldr	x0, [x0, #4064]
    1908:	f94077a2 	ldr	x2, [x29, #232]
    190c:	f9400003 	ldr	x3, [x0]
    1910:	eb030042 	subs	x2, x2, x3
    1914:	d2800003 	mov	x3, #0x0                   	// #0
    1918:	54000040 	b.eq	1920 <main+0x45c>  // b.none
    191c:	97fffba1 	bl	7a0 <__stack_chk_fail@plt>
    1920:	2a0103e0 	mov	w0, w1
    1924:	910003bf 	mov	sp, x29
    1928:	a94153f3 	ldp	x19, x20, [sp, #16]
    192c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1930:	a94363f7 	ldp	x23, x24, [sp, #48]
    1934:	a9446bf9 	ldp	x25, x26, [sp, #64]
    1938:	a94573fb 	ldp	x27, x28, [sp, #80]
    193c:	a8cf7bfd 	ldp	x29, x30, [sp], #240
    1940:	d65f03c0 	ret
    1944:	d503201f 	nop

0000000000001948 <__libc_csu_init>:
    1948:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    194c:	910003fd 	mov	x29, sp
    1950:	a90153f3 	stp	x19, x20, [sp, #16]
    1954:	90000094 	adrp	x20, 11000 <__FRAME_END__+0xf340>
    1958:	91354294 	add	x20, x20, #0xd50
    195c:	a9025bf5 	stp	x21, x22, [sp, #32]
    1960:	90000095 	adrp	x21, 11000 <__FRAME_END__+0xf340>
    1964:	913522b5 	add	x21, x21, #0xd48
    1968:	cb150294 	sub	x20, x20, x21
    196c:	2a0003f6 	mov	w22, w0
    1970:	a90363f7 	stp	x23, x24, [sp, #48]
    1974:	aa0103f7 	mov	x23, x1
    1978:	aa0203f8 	mov	x24, x2
    197c:	97fffb6f 	bl	738 <_init>
    1980:	eb940fff 	cmp	xzr, x20, asr #3
    1984:	54000160 	b.eq	19b0 <__libc_csu_init+0x68>  // b.none
    1988:	9343fe94 	asr	x20, x20, #3
    198c:	d2800013 	mov	x19, #0x0                   	// #0
    1990:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    1994:	aa1803e2 	mov	x2, x24
    1998:	91000673 	add	x19, x19, #0x1
    199c:	aa1703e1 	mov	x1, x23
    19a0:	2a1603e0 	mov	w0, w22
    19a4:	d63f0060 	blr	x3
    19a8:	eb13029f 	cmp	x20, x19
    19ac:	54ffff21 	b.ne	1990 <__libc_csu_init+0x48>  // b.any
    19b0:	a94153f3 	ldp	x19, x20, [sp, #16]
    19b4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    19b8:	a94363f7 	ldp	x23, x24, [sp, #48]
    19bc:	a8c47bfd 	ldp	x29, x30, [sp], #64
    19c0:	d65f03c0 	ret
    19c4:	d503201f 	nop

00000000000019c8 <__libc_csu_fini>:
    19c8:	d65f03c0 	ret

Disassembly of section .fini:

00000000000019cc <_fini>:
    19cc:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    19d0:	910003fd 	mov	x29, sp
    19d4:	a8c17bfd 	ldp	x29, x30, [sp], #16
    19d8:	d65f03c0 	ret
