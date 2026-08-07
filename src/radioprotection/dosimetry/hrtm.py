from radioprotection.utils import (
    read_hrtm_data,
    assign_gender_children,
    determine_breathing_rate,
    determine_inhalation_dose_coefficients,
    determine_dose
)

from radioprotection.visualization import (
    check_provided_time,
    check_or_stablish_Z_levels,
    define_X_Y_values,
    stablish_maximum_concentration,
    define_initial_plotting_parameters,
    define_Z_values,
    plot_3d,
    plot_2d,
    define_color_bar,
    plot_title,
    show_plot
)

class HRTM:
    def __init__(self, any_diffusion_advection_decay, population_type = "public", age_group = "adult", gender = "male", physical_activity = "sitting", absorption = "F", exposition_time = 30):
        self.__concentration, self.__n, self.__v, self.__species_name, self.__diffusion_coefficient , self.__time = any_diffusion_advection_decay.provide_variables_hrtm()
        self.__population_type = population_type
        self.__age_group = age_group
        self.__gender = gender
        self.__physical_activity = physical_activity
        self.__absorption = absorption
        self.__exposition_time = exposition_time

        self.__data = read_hrtm_data()

    def effective_dose_commitment(self):
        self.__gender = assign_gender_children(self.__gender, self.__age_group)

        breathing_rate = determine_breathing_rate(self.__data, self.__age_group, self.__gender, self.__physical_activity)

        inhalation_dose_coefficients = determine_inhalation_dose_coefficients(self.__data, self.__population_type, self.__species_name, self.__absorption, self.__age_group)

        self.__dose = determine_dose(self.__concentration, breathing_rate, self.__exposition_time, inhalation_dose_coefficients)

        return self.__dose


    def spatial_visualization(self, visualization_type = "3d", vertical_axis = "z", levels = None, time = None):

        time = check_provided_time(time, self.__time, self.__dose)

        check_or_stablish_Z_levels(vertical_axis, levels)

        X, Y, aux_axis = define_X_Y_values(vertical_axis, self.__n)

        concentration_max = stablish_maximum_concentration(time, self.__dose)

        fig, norm = define_initial_plotting_parameters()

        for i, level in enumerate(levels):

            Z = define_Z_values(self.__dose, vertical_axis, concentration_max, time, level)

            if (visualization_type=="3d"):
                plot_3d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, concentration_max, vertical_axis_label=fr"Dose ($\times$ ({concentration_max:.2e})$^{{-1}}$ Sv)", iteration = i)

            elif (visualization_type == "2d"):
                plot_2d(X, Y, Z, fig, norm, vertical_axis, level, aux_axis, i)

            else:
                raise ValueError("The provided string for visualization type is not valid.")

        define_color_bar(fig, norm, concentration_max, vertical_axis_label= fr"Dose ($\times$ ({concentration_max:.2e})$^{{-1}}$ Sv)")

        plot_title(fig, f"Radioisotope = {self.__species_name} | Visualization type = {visualization_type} | Instant = {time} s | Wind speed = {self.__v} | Diffusion coefficient = {self.__diffusion_coefficient}\nPopulation Type = {self.__population_type} | Age Group = {self.__age_group} | Gender = {self.__gender} | Physical Activity = {self.__physical_activity} | Absorption = {self.__absorption} | Exposition time = {self.__exposition_time}")

        show_plot()