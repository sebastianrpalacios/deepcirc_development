module m0x79 (input in2, in1, in3, output out);

	wire \$n14_0;
	wire \$n10_0;
	wire \$n11_0;
	wire \$n12_0;
	wire \$n13_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n8_1;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	not (\$n9_0, \$n8_0);
	not (\$n5_0, in2);
	not (\$n6_0, in3);
	not (\$n7_0, in1);
	nor (\$n10_0, \$n6_0, \$n5_0);
	nor (\$n11_0, \$n10_0, \$n7_0);
	nor (\$n8_0, in2, in3);
	nor (\$n8_1, in2, in3);
	not (\$n12_0, \$n11_0);
	nor (\$n13_0, \$n12_0, \$n8_1);
	nor (\$n14_0, in1, \$n9_0);
	nor (out, \$n14_0, \$n13_0);

endmodule
