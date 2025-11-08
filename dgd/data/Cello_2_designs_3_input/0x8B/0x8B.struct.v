module m0x8B (input in2, in1, in3, output out);

	wire \$n8_0;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in2);
	not (\$n6_0, in3);
	nor (\$n7_0, in2, \$n6_0);
	nor (\$n8_0, in1, \$n5_0);
	nor (out, \$n8_0, \$n7_0);

endmodule
