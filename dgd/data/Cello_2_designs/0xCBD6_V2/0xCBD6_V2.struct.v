module m0xCBD6 (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n11_0;
	wire \$n11_1;
	wire \$n20_0;
	wire \$n14_0;
	wire \$n14_1;
	wire \$n15_0;
	wire \$n9_0;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n13_0;
	wire \$n7_0;
	wire \$n18_0;
	wire \$n19_0;
	wire \$n16_0;
	wire \$n17_0;
	wire \$n6_0;

	nor (\$n17_0, in2, in3);
	nor (\$n18_0, \$n14_0, \$n17_0);
	not (\$n19_0, \$n18_0);
	not (\$n7_0, in3);
	nor (\$n13_0, \$n7_0, in4);
	not (\$n9_0, in1);
	nor (\$n14_0, \$n9_0, \$n13_0);
	nor (\$n14_1, \$n9_0, \$n13_0);
	not (\$n6_0, in4);
	not (\$n8_0, in2);
	nor (\$n10_0, in3, \$n6_0);
	nor (\$n11_0, \$n10_0, \$n8_0);
	nor (\$n11_1, \$n10_0, \$n8_0);
	not (\$n12_0, \$n11_1);
	not (\$n15_0, \$n14_1);
	nor (\$n16_0, \$n15_0, \$n12_0);
	nor (\$n20_0, \$n11_0, \$n19_0);
	nor (out, \$n20_0, \$n16_0);

endmodule
