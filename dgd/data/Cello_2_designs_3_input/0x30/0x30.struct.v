module m0x30 (input in2, in1, in3, output out);

	wire \$n5_0;

	not (\$n5_0, in2);
	nor (out, in1, \$n5_0);

endmodule
