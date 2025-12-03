module layer2 (
    input clk,
    input rst_n,
    // 3 Input Channels (12-bit signed data from Layer 1)
    input signed [11:0] in0, in1, in2,
    input valid_in,
    // Outputs: 3 Filters, 12-bit signed data (after pooling)
    output reg signed [11:0] out0, out1, out2,
    output reg valid_out
);

    // --- 1. WEIGHTS WIRE (3 Filters x 3 Channels x 9 Kernel = 81 Weights) ---
    wire signed [7:0] weight [0:80];

    // ------------------------------------------------------------------------
    // PASTE YOUR 'rom_conv2.txt' CONTENT BELOW
    // ------------------------------------------------------------------------
    
    // [PASTE HERE]
assign weight[0] = 8'hac;
assign weight[1] = 8'hd2;
assign weight[2] = 8'hc8;
assign weight[3] = 8'hb0;
assign weight[4] = 8'h1b;
assign weight[5] = 8'h13;
assign weight[6] = 8'h17;
assign weight[7] = 8'hb6;
assign weight[8] = 8'hfc;
assign weight[9] = 8'h3a;
assign weight[10] = 8'h07;
assign weight[11] = 8'hf9;
assign weight[12] = 8'h21;
assign weight[13] = 8'hee;
assign weight[14] = 8'h3a;
assign weight[15] = 8'he1;
assign weight[16] = 8'hfa;
assign weight[17] = 8'hfe;
assign weight[18] = 8'h12;
assign weight[19] = 8'he6;
assign weight[20] = 8'hcb;
assign weight[21] = 8'h16;
assign weight[22] = 8'h0d;
assign weight[23] = 8'h0f;
assign weight[24] = 8'h07;
assign weight[25] = 8'h39;
assign weight[26] = 8'h26;
assign weight[27] = 8'h05;
assign weight[28] = 8'hcf;
assign weight[29] = 8'h0c;
assign weight[30] = 8'h04;
assign weight[31] = 8'h27;
assign weight[32] = 8'h20;
assign weight[33] = 8'heb;
assign weight[34] = 8'hee;
assign weight[35] = 8'hc1;
assign weight[36] = 8'h45;
assign weight[37] = 8'h92;
assign weight[38] = 8'hc3;
assign weight[39] = 8'h05;
assign weight[40] = 8'hba;
assign weight[41] = 8'hbc;
assign weight[42] = 8'h43;
assign weight[43] = 8'h27;
assign weight[44] = 8'hbc;
assign weight[45] = 8'h7f;
assign weight[46] = 8'hff;
assign weight[47] = 8'h1a;
assign weight[48] = 8'h57;
assign weight[49] = 8'hc6;
assign weight[50] = 8'he7;
assign weight[51] = 8'h29;
assign weight[52] = 8'h08;
assign weight[53] = 8'h2a;
assign weight[54] = 8'h74;
assign weight[55] = 8'h09;
assign weight[56] = 8'h2f;
assign weight[57] = 8'h05;
assign weight[58] = 8'hdf;
assign weight[59] = 8'he7;
assign weight[60] = 8'he8;
assign weight[61] = 8'hf6;
assign weight[62] = 8'hde;
assign weight[63] = 8'he8;
assign weight[64] = 8'hfe;
assign weight[65] = 8'h30;
assign weight[66] = 8'he0;
assign weight[67] = 8'hff;
assign weight[68] = 8'hef;
assign weight[69] = 8'h38;
assign weight[70] = 8'h0b;
assign weight[71] = 8'h0f;
assign weight[72] = 8'h41;
assign weight[73] = 8'he2;
assign weight[74] = 8'hff;
assign weight[75] = 8'hf2;
assign weight[76] = 8'hd9;
assign weight[77] = 8'hbc;
assign weight[78] = 8'hbb;
assign weight[79] = 8'h00;
assign weight[80] = 8'hd3;
    
    // ------------------------------------------------------------------------

    // --- 2. LINE BUFFERS (13x13 Input, 12-bit) ---
    // We need 3 separate buffers, one for each input channel.
    
    wire [11:0] w0_00, w0_01, w0_02, w0_10, w0_11, w0_12, w0_20, w0_21, w0_22;
    wire [11:0] w1_00, w1_01, w1_02, w1_10, w1_11, w1_12, w1_20, w1_21, w1_22;
    wire [11:0] w2_00, w2_01, w2_02, w2_10, w2_11, w2_12, w2_20, w2_21, w2_22;
    wire win_valid;

    // Buffer 0 (Channel 0)
    conv_buf #(.WIDTH(13), .DATA_BITS(12)) lb0 (
        .clk(clk), .rst_n(rst_n), .data_in(in0), .valid_in(valid_in),
        .w00(w0_00), .w01(w0_01), .w02(w0_02), .w10(w0_10), .w11(w0_11), .w12(w0_12), .w20(w0_20), .w21(w0_21), .w22(w0_22),
        .valid_out(win_valid) // Master valid signal
    );
    // Buffer 1 (Channel 1)
    conv_buf #(.WIDTH(13), .DATA_BITS(12)) lb1 (
        .clk(clk), .rst_n(rst_n), .data_in(in1), .valid_in(valid_in),
        .w00(w1_00), .w01(w1_01), .w02(w1_02), .w10(w1_10), .w11(w1_11), .w12(w1_12), .w20(w1_20), .w21(w1_21), .w22(w1_22),
        .valid_out() // Slave
    );
    // Buffer 2 (Channel 2)
    conv_buf #(.WIDTH(13), .DATA_BITS(12)) lb2 (
        .clk(clk), .rst_n(rst_n), .data_in(in2), .valid_in(valid_in),
        .w00(w2_00), .w01(w2_01), .w02(w2_02), .w10(w2_10), .w11(w2_11), .w12(w2_12), .w20(w2_20), .w21(w2_21), .w22(w2_22),
        .valid_out() // Slave
    );

    // --- 3. CONVOLUTION (9 Parallel Calculators) ---
    // Architecture: 
    // Filter 0 Sum = (In0 * W0_0) + (In1 * W0_1) + (In2 * W0_2)
    // Filter 1 Sum = (In0 * W1_0) + ...
    
    wire signed [31:0] f0_c0, f0_c1, f0_c2;
    wire signed [31:0] f1_c0, f1_c1, f1_c2;
    wire signed [31:0] f2_c0, f2_c1, f2_c2;
    wire calc_valid;

    // FILTER 0 (Weights 0-26)
    conv_calc #(.DATA_BITS(12)) c0_0 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w0_00), .p01(w0_01), .p02(w0_02), .p10(w0_10), .p11(w0_11), .p12(w0_12), .p20(w0_20), .p21(w0_21), .p22(w0_22), .k00(weight[0]), .k01(weight[1]), .k02(weight[2]), .k10(weight[3]), .k11(weight[4]), .k12(weight[5]), .k20(weight[6]), .k21(weight[7]), .k22(weight[8]), .result(f0_c0), .valid_out(calc_valid));
    conv_calc #(.DATA_BITS(12)) c0_1 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w1_00), .p01(w1_01), .p02(w1_02), .p10(w1_10), .p11(w1_11), .p12(w1_12), .p20(w1_20), .p21(w1_21), .p22(w1_22), .k00(weight[9]), .k01(weight[10]), .k02(weight[11]), .k10(weight[12]), .k11(weight[13]), .k12(weight[14]), .k20(weight[15]), .k21(weight[16]), .k22(weight[17]), .result(f0_c1), .valid_out());
    conv_calc #(.DATA_BITS(12)) c0_2 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w2_00), .p01(w2_01), .p02(w2_02), .p10(w2_10), .p11(w2_11), .p12(w2_12), .p20(w2_20), .p21(w2_21), .p22(w2_22), .k00(weight[18]), .k01(weight[19]), .k02(weight[20]), .k10(weight[21]), .k11(weight[22]), .k12(weight[23]), .k20(weight[24]), .k21(weight[25]), .k22(weight[26]), .result(f0_c2), .valid_out());

    // FILTER 1 (Weights 27-53)
    conv_calc #(.DATA_BITS(12)) c1_0 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w0_00), .p01(w0_01), .p02(w0_02), .p10(w0_10), .p11(w0_11), .p12(w0_12), .p20(w0_20), .p21(w0_21), .p22(w0_22), .k00(weight[27]), .k01(weight[28]), .k02(weight[29]), .k10(weight[30]), .k11(weight[31]), .k12(weight[32]), .k20(weight[33]), .k21(weight[34]), .k22(weight[35]), .result(f1_c0), .valid_out());
    conv_calc #(.DATA_BITS(12)) c1_1 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w1_00), .p01(w1_01), .p02(w1_02), .p10(w1_10), .p11(w1_11), .p12(w1_12), .p20(w1_20), .p21(w1_21), .p22(w1_22), .k00(weight[36]), .k01(weight[37]), .k02(weight[38]), .k10(weight[39]), .k11(weight[40]), .k12(weight[41]), .k20(weight[42]), .k21(weight[43]), .k22(weight[44]), .result(f1_c1), .valid_out());
    conv_calc #(.DATA_BITS(12)) c1_2 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w2_00), .p01(w2_01), .p02(w2_02), .p10(w2_10), .p11(w2_11), .p12(w2_12), .p20(w2_20), .p21(w2_21), .p22(w2_22), .k00(weight[45]), .k01(weight[46]), .k02(weight[47]), .k10(weight[48]), .k11(weight[49]), .k12(weight[50]), .k20(weight[51]), .k21(weight[52]), .k22(weight[53]), .result(f1_c2), .valid_out());

    // FILTER 2 (Weights 54-80)
    conv_calc #(.DATA_BITS(12)) c2_0 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w0_00), .p01(w0_01), .p02(w0_02), .p10(w0_10), .p11(w0_11), .p12(w0_12), .p20(w0_20), .p21(w0_21), .p22(w0_22), .k00(weight[54]), .k01(weight[55]), .k02(weight[56]), .k10(weight[57]), .k11(weight[58]), .k12(weight[59]), .k20(weight[60]), .k21(weight[61]), .k22(weight[62]), .result(f2_c0), .valid_out());
    conv_calc #(.DATA_BITS(12)) c2_1 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w1_00), .p01(w1_01), .p02(w1_02), .p10(w1_10), .p11(w1_11), .p12(w1_12), .p20(w1_20), .p21(w1_21), .p22(w1_22), .k00(weight[63]), .k01(weight[64]), .k02(weight[65]), .k10(weight[66]), .k11(weight[67]), .k12(weight[68]), .k20(weight[69]), .k21(weight[70]), .k22(weight[71]), .result(f2_c1), .valid_out());
    conv_calc #(.DATA_BITS(12)) c2_2 (.clk(clk), .rst_n(rst_n), .valid_in(win_valid), .p00(w2_00), .p01(w2_01), .p02(w2_02), .p10(w2_10), .p11(w2_11), .p12(w2_12), .p20(w2_20), .p21(w2_21), .p22(w2_22), .k00(weight[72]), .k01(weight[73]), .k02(weight[74]), .k10(weight[75]), .k11(weight[76]), .k12(weight[77]), .k20(weight[78]), .k21(weight[79]), .k22(weight[80]), .result(f2_c2), .valid_out());

    // --- 4. SUMMATION + ReLU + POOLING ---
    // Conv Output size is 11x11 (13 - 3 + 1).
    reg [9:0] x_cnt, y_cnt;
    reg signed [31:0] sum0, sum1, sum2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= 0; y_cnt <= 0;
            valid_out <= 0;
            out0 <= 0; out1 <= 0; out2 <= 0;
        end else if (calc_valid) begin
            
            // 1. Sum Channels
            sum0 = f0_c0 + f0_c1 + f0_c2;
            sum1 = f1_c0 + f1_c1 + f1_c2;
            sum2 = f2_c0 + f2_c1 + f2_c2;

            // 2. Max Pooling (Subsample Top-Left of 2x2)
            if (x_cnt[0] == 0 && y_cnt[0] == 0) begin
                valid_out <= 1;
                // 3. ReLU + Scaling (Divide by 16 = >> 4)
                // We shift right by 4 to bring 32-bit accumulated value back to reasonable 12-bit range
                // Adjust shift amount [15:4] if output is too dark/bright.
                out0 <= (sum0 > 0) ? sum0[15:4] : 12'd0;
                out1 <= (sum1 > 0) ? sum1[15:4] : 12'd0;
                out2 <= (sum2 > 0) ? sum2[15:4] : 12'd0;
            end else begin
                valid_out <= 0;
            end

            // 3. Counters (0 to 10) for 11x11 image
            if (x_cnt == 10) begin
                x_cnt <= 0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end
        end else begin
            valid_out <= 0;
        end
    end

endmodule