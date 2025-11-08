module m0x46 (input in2, in1, in3, output out);

	wire \$n10_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;
	wire \$n5_1;

	not (\$n6_0, in3);
	not (\$n5_0, in2);
	not (\$n5_1, in2);
	not (\$n7_0, in1);
	nor (\$n8_0, \$n7_0, \$n5_1);
	nor (\$n9_0, \$n8_0, in3);
	nor (\$n10_0, \$n6_0, \$n5_0);
	nor (out, \$n10_0, \$n9_0);

endmodule
