module m0x7E (input in2, in1, in3, output out);

	wire \$new_n9__0;
	wire \$new_n8__0;
	wire \$new_n7__0;
	wire \$new_n6__0;
	wire \$new_n10__0;
	wire \$new_n11__0;
	wire \$new_n12__0;
	wire \$new_n5__0;

	not (\$new_n6__0, in3);
	not (\$new_n7__0, in1);
	not (\$new_n5__0, in2);
	nor (\$new_n8__0, \$new_n7__0, \$new_n6__0);
	nor (\$new_n10__0, in1, in3);
	nor (\$new_n11__0, \$new_n10__0, in2);
	nor (\$new_n9__0, \$new_n8__0, \$new_n5__0);
	nor (\$new_n12__0, \$new_n9__0, \$new_n11__0);
	not (out, \$new_n12__0);

endmodule
