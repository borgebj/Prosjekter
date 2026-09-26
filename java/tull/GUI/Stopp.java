package tull.GUI;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.Pane;
import javafx.scene.control.Button;
import javafx.event.ActionEvent;

public class Stopp extends Application {

    static class StoppBehandler implements EventHandler<ActionEvent> {
        @Override
        public void handle(ActionEvent e) {
            Platform.exit();
        }
    }

    @Override
    public void start(Stage teater) {
        StoppBehandler stopp = new StoppBehandler();
        Button stoppKnapp = new Button("Stopp");
        stoppKnapp.setOnAction(stopp);

        Pane kulisser = new Pane();
        kulisser.getChildren().add(stoppKnapp);
        Scene scene = new Scene(kulisser);
        teater.setTitle("Klikk for å stoppe");
        teater.setScene(scene);
        teater.show();
    }
}