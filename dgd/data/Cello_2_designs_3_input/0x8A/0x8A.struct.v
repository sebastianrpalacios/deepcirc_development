module m0x8A (input in2, in1, in3, output out);

	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in2);
	nor (\$n6_0, in1, \$n5_0);
	nor (out, \$n6_0, in3);

endmodule
