# Creating an app package
1) Build the application as usual within Unity.
2) In the Visual Studio project, right-click the project in the Solution Explorer and select `Publish -> Create App Packages...`

![Menu](docs/images/publish_screen.png)

3) In the app package window, select `Sideloading`, uncheck the `Enable automatic updates` box, and click `Next`.

![Distribution screen](docs/images/distribution_method.PNG)

4) Ensure the app certificate is signed and click `Next`. The default certificate settings are OK.

![Signing screen](docs/images/signing_method.PNG)

5) Choose the app's output location and ensure the ARM64 architecture is selected. Click `Create` to create the package.

![Package screen](docs/images/package_configuration.PNG)

# Installing package via the device portal
Follow the installation steps from the [Windows Device Portal Documentation](https://docs.microsoft.com/en-us/windows/mixed-reality/develop/advanced-concepts/using-the-windows-device-portal#installing-an-app).

Select the .msix file in the ARM64 folder in the app package created previously.
Note you will need to remove any previously installed apps of the same name. This can be done in the device portal Apps view by selecting the app in the Installed apps window and clicking `Remove`.

![Device portal install](docs/images/device_portal_install.PNG)
