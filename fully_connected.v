module fully_connected (
    input clk,
    input rst_n,
    input valid_in,
    // 3 Inputs from Layer 3 (12-bit signed)
    input signed [11:0] in0, in1, in2,
    // Output: 12-bit Score (Streamed: Class 0 then Class 1)
    output reg signed [11:0] data_out,
    output reg valid_out
);

    // --- 1. WEIGHTS WIRE (3 Inputs x 2 Classes = 6 Weights) ---
    wire signed [7:0] weight [0:5];

    // ------------------------------------------------------------------------
    // PASTE YOUR 'rom_fc.txt' CONTENT BELOW
    // ------------------------------------------------------------------------
    
    // [PASTE HERE]
assign weight[0] = 8'hbe;
assign weight[1] = 8'h7f;
assign weight[2] = 8'h19;
assign weight[3] = 8'h13;
assign weight[4] = 8'hcc;
assign weight[5] = 8'hd3;
    
    // ------------------------------------------------------------------------

    // --- ROBUST STATE MACHINE ---
    // This latches the input pulse from Layer 3 and processes it over 2 cycles.
    
    reg [1:0] process_cnt; // 0=Idle, 1=Class0, 2=Class1
    reg signed [11:0] latched_in0, latched_in1, latched_in2;
    reg signed [19:0] acc; // Accumulator

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
            data_out <= 0;
            process_cnt <= 0;
            latched_in0 <= 0;
            latched_in1 <= 0;
            latched_in2 <= 0;
            acc <= 0;
        end else begin
            // 1. Trigger: Latch data on valid pulse
            if (valid_in) begin
                latched_in0 <= in0;
                latched_in1 <= in1;
                latched_in2 <= in2;
                process_cnt <= 1; // Start processing
                valid_out <= 0;
            end 
            
            // 2. Cycle 1: Calculate & Output Class 0 (Non-Smoking)
            else if (process_cnt == 1) begin
                acc = (latched_in0 * weight[0]) + (latched_in1 * weight[1]) + (latched_in2 * weight[2]);
                
                // Scale (Divide by 128 = shift right 7)
                // Check for saturation/overflow if needed, but simple shift is usually okay here
                data_out <= acc[18:7]; 
                
                valid_out <= 1;
                process_cnt <= 2;
            end
            
            // 3. Cycle 2: Calculate & Output Class 1 (Smoking)
            else if (process_cnt == 2) begin
                acc = (latched_in0 * weight[3]) + (latched_in1 * weight[4]) + (latched_in2 * weight[5]);
                
                data_out <= acc[18:7];
                
                valid_out <= 1;
                process_cnt <= 0; // Done, return to idle
            end
            
            // 4. Idle State
            else begin
                valid_out <= 0;
            end
        end
    end

endmodule