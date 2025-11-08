module m0x22 (input in2, in1, in3, output out);

	wire \$n5_0;

	not (\$n5_0, in2);
	nor (out, \$n5_0, in3);

endmodule
