#include "Labs/5-Visualization/tasks.h"

#include <numbers>
#include<climits>
#include<cmath>
#include<cstring>
using VCX::Labs::Common::ImageRGB;
namespace VCX::Labs::Visualization {

    struct CoordinateStates {
        // your code here
        std::vector<Car>data;
        float max_values[7];
        float min_values[7];
        CoordinateStates(const std::vector<Car> & data):data(data)
        {
            for (int i = 0; i < 7; i++)
            {
                max_values[i]=INT_MIN;
                min_values[i]=INT_MAX;
            }
            for (const auto& car_ : data)
            {
                max_values[0] = std::max(max_values[0], float(car_.cylinders));
                min_values[0] = std::min(min_values[0], float(car_.cylinders));
                max_values[1] = std::max(max_values[1], float(car_.displacement));
                min_values[1] = std::min(min_values[1], float(car_.displacement));
                max_values[2] = std::max(max_values[2], float(car_.weight));
                min_values[2] = std::min(min_values[2], float(car_.weight));
                max_values[3] = std::max(max_values[3], float(car_.horsepower));
                min_values[3] = std::min(min_values[3], float(car_.horsepower));
                max_values[4] = std::max(max_values[4], float(car_.acceleration));
                min_values[4] = std::min(min_values[4], float(car_.acceleration));
                max_values[5] = std::max(max_values[5], float(car_.mileage));
                min_values[5] = std::min(min_values[5], float(car_.mileage));
                max_values[6] = std::max(max_values[6], float(car_.year));
                min_values[6] = std::min(min_values[6], float(car_.year));
            }

        }

    };

    float linear_interpolation(float val_,float min_, float max_, float down_, float up_)
    {
        // down_ is the max_y in the pixel_coordinates // up_ is the min_y in the pixel_coordinates 
        // min_ is the minvalue of the attribute //max_ is the maxvalue of the attribute
        return (val_ - min_) / (max_ - min_)*(down_-up_)+up_;
    }

    bool first_draw = false;
    int  select_bars[7] = {0,0,0,0,0,0,0};
    float select_bars_ypos[7][2];

    bool PaintParallelCoordinates(Common::ImageRGB & input, InteractProxy const & proxy, std::vector<Car> const & data, bool force) {
        // your code here
        // for example: 
        //   static CoordinateStates states(data);
        //   SetBackGround(input, glm::vec4(1));
        //   ...
        
        int  mouse_mode     = 0;

        glm::vec2 mouse_first_pos(0.0f, 0.0f);
        glm::vec2 mouse_second_pos(0.0f, 0.0f);
        if (proxy.IsHovering())
        {
            if (proxy.IsClicking(true))
            {
                mouse_first_pos=proxy.MousePos();
                mouse_mode = 1;
            }
            else if (proxy.IsDragging(true))
            {
                mouse_mode = 2;
                mouse_first_pos = proxy.DraggingStartPoint();
                mouse_second_pos = proxy.MousePos();
            }
            else if (proxy.IsClicking(false))
            {
                first_draw=false;//refresh
                memset(select_bars, 0, sizeof(select_bars));
            }
        }
        // first_draw over
        if (first_draw && mouse_mode==0)return false; 
       
        // initialize Data & Background
        static CoordinateStates states(data);
        SetBackGround(input, glm::vec4(1));

        // data process
        float padding_percent = 0.1;
        int max_vals[7];
        int min_vals[7];
        float   coor_x[7];
        std::string properties_tag[7] = { "cylinders", "displacement", "weight", "horsepower", "acceleration(0-60mph)", "mileage", "year" };
        for (int i = 0; i < 7; i++)
        {
            int distance = int((states.max_values[i] - states.min_values[i])*padding_percent)+1;
            max_vals[i]  = states.max_values[i] + distance;
            min_vals[i]  = states.min_values[i] - distance;
            coor_x[i]    = (0.85 / 6)*i + 0.075;
        }

        // draw coordinates

        float bar_up=0.15;
        float bar_down = 0.9;
        float bar_width = 0.04;
        for (int i = 0; i < 7; i++)
        {
            // Draw vertical line for each bar
            DrawLine(input, glm::vec4 { 0, 0, 0, 1 }, glm::vec2 { coor_x[i], bar_up }, glm::vec2 { coor_x[i], bar_down }, 3);
            DrawFilledRect(input, glm::vec4 { 0, 0.5, 0.5, 0.3 }, glm::vec2 { coor_x[i] - bar_width/2, bar_up }, glm::vec2 { bar_width, bar_down-bar_up });
            
            // Draw tag of each bar
            // properties tag
            PrintText(input, glm::vec4 { 0, 0, 0, 1 }, glm::vec2 { coor_x[i], 0.09 },0.015, properties_tag[i]);
            // max tag
            PrintText(input, glm::vec4 { 0, 0, 0, 1 }, glm::vec2 { coor_x[i], 0.125 },0.03, std::to_string(max_vals[i]));
            // min tag
            PrintText(input, glm::vec4 { 0, 0, 0, 1 }, glm::vec2 { coor_x[i], 0.925 },0.03, std::to_string(min_vals[i]));
        }
        
        int select_bar = -1;
        if (mouse_mode != 0)
        {
            for (int i = 0; i < 7; i++)
            {
                if (abs(mouse_first_pos.x - coor_x[i]) < bar_width / 2)
                {
                    mouse_first_pos.y  = std::max(bar_up, mouse_first_pos.y);
                    mouse_first_pos.y  = std::min(bar_down, mouse_first_pos.y);
                    mouse_second_pos.y = std::max(bar_up, mouse_second_pos.y);
                    mouse_second_pos.y = std::min(bar_down, mouse_second_pos.y);
                    
                    select_bar = i;
                    if (mouse_mode == 2)
                    {
                        select_bars[i]         = 1;
                        select_bars_ypos[i][0] = mouse_first_pos.y;
                        select_bars_ypos[i][1] = mouse_second_pos.y;
                    }
                    break;
                }
            }
        }

        float origin_alpha = 0.8;
        float lower_alpha  = 0.2;
        if (first_draw == false) {
            int   data_num = states.data.size();
            float tmp_yi1, tmp_yi2;
            for (int i = 0; i < data_num; i++) {
                float tmp_data[7] = { (float) states.data[i].cylinders, states.data[i].displacement, states.data[i].weight, states.data[i].horsepower, states.data[i].acceleration, states.data[i].mileage, (float) states.data[i].year };
                tmp_yi1           = linear_interpolation(tmp_data[0], min_vals[0], max_vals[0], bar_down, bar_up);
                for (int j = 1; j < 7; j++) {
                    tmp_yi2 = linear_interpolation(tmp_data[j], min_vals[j], max_vals[j], bar_down, bar_up);
                    DrawLine(input, glm::vec4(1.0 - 0.9 * i / data_num, 0.2, 0.9 * i / data_num, origin_alpha), glm::vec2 { coor_x[j - 1], tmp_yi1 }, glm::vec2 { coor_x[j], tmp_yi2 }, 1.2);
                    tmp_yi1 = tmp_yi2;
                }
            }
            first_draw = true;
            return true;
        } 
        else if (select_bar!=-1)
        {

           
            
            if (mouse_mode == 1)
            {
                DrawFilledRect(input, glm::vec4 { 1.0, 0.2, 0.1, 0.7 }, glm::vec2 { coor_x[select_bar] - bar_width / 2, bar_up }, glm::vec2 { bar_width, bar_down - bar_up });
                memset(select_bars, 0, sizeof(select_bars));
                int   data_num = states.data.size();
                float tmp_yi1, tmp_yi2;
                for (int i = 0; i < data_num; i++) {
                    
                    float tmp_data[7] = { (float) states.data[i].cylinders, states.data[i].displacement, states.data[i].weight, states.data[i].horsepower, states.data[i].acceleration, states.data[i].mileage, (float) states.data[i].year };
                    float color_k     = (tmp_data[select_bar] - min_vals[select_bar]) / (max_vals[select_bar] - min_vals[select_bar]);

                    tmp_yi1           = linear_interpolation(tmp_data[0], min_vals[0], max_vals[0], bar_down, bar_up);
                    for (int j = 1; j < 7; j++) {
                        tmp_yi2 = linear_interpolation(tmp_data[j], min_vals[j], max_vals[j], bar_down, bar_up);
                        DrawLine(input, glm::vec4(0.9 * color_k, 0.2, 1.0-0.9 * color_k, origin_alpha), glm::vec2 { coor_x[j - 1], tmp_yi1 }, glm::vec2 { coor_x[j], tmp_yi2 }, 1.2);
                        tmp_yi1 = tmp_yi2;
                    }
                }
                return true;
            }
            else if (mouse_mode == 2)
            {

                for (int i = 0; i < 7; i++)
                {
                    if (select_bars[i] == 0) continue;
                    DrawFilledRect(input, glm::vec4 { 1.0, 0.2, 0.1, 0.7 }, glm::vec2 { coor_x[i] - bar_width / 2, bar_up }, glm::vec2 { bar_width, bar_down - bar_up });
                }

                int   data_num = states.data.size();
                float tmp_yi1, tmp_yi2;
                for (int i = 0; i < data_num; i++) {
                    float tmp_data[7] = { (float) states.data[i].cylinders, states.data[i].displacement, states.data[i].weight, states.data[i].horsepower, states.data[i].acceleration, states.data[i].mileage, (float) states.data[i].year };
                    float color_k     = (tmp_data[select_bar] - min_vals[select_bar]) / (max_vals[select_bar] - min_vals[select_bar]);


                    float tmp_alpha = origin_alpha;
                    

                    for (int j = 0; j < 7; j++)
                    {
                        if (select_bars[j] == 0) continue;
                        float select_y = linear_interpolation(tmp_data[j], min_vals[j], max_vals[j], bar_down, bar_up);
                        if (select_y < std::min(select_bars_ypos[j][0], select_bars_ypos[j][1]) || select_y > std::max(select_bars_ypos[j][0], select_bars_ypos[j][1])) 
                        {
                            tmp_alpha = lower_alpha;
                            break;
                        }
                    }
                    

                    tmp_yi1 = linear_interpolation(tmp_data[0], min_vals[0], max_vals[0], bar_down, bar_up);
                    for (int j = 1; j < 7; j++) {
                        tmp_yi2 = linear_interpolation(tmp_data[j], min_vals[j], max_vals[j], bar_down, bar_up);
                        DrawLine(input, glm::vec4(0.9 * color_k, 0.2, 1.0 - 0.9 * color_k, tmp_alpha), glm::vec2 { coor_x[j - 1], tmp_yi1 }, glm::vec2 { coor_x[j], tmp_yi2 }, 1.2);
                        tmp_yi1 = tmp_yi2;
                    }
                }

                for (int j = 0; j < 7; j++) {
                    if (select_bars[j] == 0) continue;
                    DrawFilledRect(input, glm::vec4 { 1.0, 0.2, 0.1, 0.9 }, glm::vec2 { coor_x[j] - bar_width / 2, std::min(select_bars_ypos[j][0], select_bars_ypos[j][1]) }, glm::vec2 { bar_width, std::max(select_bars_ypos[j][0], select_bars_ypos[j][1]) - std::min(select_bars_ypos[j][0], select_bars_ypos[j][1]) });
                    
                    float bar_height  = bar_down - bar_up;
                    int   val_distance = max_vals[j] - min_vals[j];
                    int   up_boundary   = max_vals[j]-(std::min(select_bars_ypos[j][0], select_bars_ypos[j][1]) - bar_up) / bar_height * (max_vals[j] - min_vals[j]);
                    int   down_boundary = max_vals[j]-(std::max(select_bars_ypos[j][0], select_bars_ypos[j][1]) - bar_up) / bar_height * (max_vals[j] - min_vals[j]);
                    // up boundary
                    PrintText(input, glm::vec4 { 0, 0, 0, 1 }, glm::vec2 { coor_x[j], std::min(select_bars_ypos[j][0], select_bars_ypos[j][1])-0.02 }, 0.02, std::to_string(up_boundary));
                    // down boundary
                    PrintText(input, glm::vec4 { 0, 0, 0, 1 }, glm::vec2 { coor_x[j], std::max(select_bars_ypos[j][0], select_bars_ypos[j][1])+0.02 }, 0.02, std::to_string(down_boundary));
                }
                

                return true;
            }
        }

        return false;

    }

    void LIC(ImageRGB & output, Common::ImageRGB const & noise, VectorField2D const & field, int const & step) {
        // your code here
        int width=noise.GetSizeX();
        int height=noise.GetSizeY();
        SetBackGround(output, glm::vec4(0));
        float t = 0;

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // Forward
                float y =i;
                float x = j;
                glm::vec3 forward_sum   = {0.0f,0.0f,0.0f};
                float forward_total=0;

                for (int k = 0; k < step; k++)
                {
                    float dx = field.At(int(x), int(y))[0];
                    float dy = field.At(int(x), int(y))[1];
                    float dt_x=0;
                    float dt_y=0;
                    float dt   = 0;

                    if (dy > 0)dt_y = (floor(y)+1-y)/dy;
                    else if (dy < 0) dt_y = -(y+1-ceil(y))/dy;

                    if (dx > 0) dt_x = (floor(x) + 1 - x) / dx;
                    else if (dx < 0) dt_x = -(x + 1 - ceil(x)) / dx;

                    if (dx == 0 && dy == 0) dt = 0;
                    else dt = std::min(dt_x, dt_y);

                    x = std::min(std::max(x + dx * dt, float(0)), float(width - 1));
                    y = std::min(std::max(y + dy * dt, float(0)), float(height - 1));

                    float weight = pow(cos(t + 0.46 * (float) k), 2);
                    forward_sum += noise.At(int(x), int(y)) * weight;
                    forward_total += weight;

                }

                // Backward
                y  = i;
                x  = j;
                glm::vec3 backward_sum   = { 0.0f, 0.0f,0.0f};
                float     backward_total = 0;

                for (int k = 1; k < step; k++) {
                    float dx   = -field.At(int(x), int(y))[0];
                    float dy   = -field.At(int(x), int(y))[1];
                    float dt_x = 0;
                    float dt_y = 0;
                    float dt   = 0;

                    if (dy > 0) dt_y = (floor(y) + 1 - y) / dy;
                    else if (dy < 0) dt_y = -(y + 1 - ceil(y)) / dy;

                    if (dx > 0) dt_x = (floor(x) + 1 - x) / dx;
                    else if (dx < 0) dt_x = -(x + 1 - ceil(x)) / dx;

                    if (dx == 0 && dy == 0) dt = 0;
                    else dt = std::min(dt_x, dt_y);

                    x = std::min(std::max(x + dx * dt, float(0)), float(width - 1));
                    y = std::min(std::max(y + dy * dt, float(0)), float(height - 1));

                    float weight = pow(cos(t + 0.46 * (float)k), 2);
                    backward_sum += noise.At(int(x), int(y)) * weight;
                    backward_total += weight;
                }

                output.At(j,i) = (forward_sum + backward_sum) / (forward_total + backward_total);
            }
        }
    }
}; // namespace VCX::Labs::Visualization