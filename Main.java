// Abstract class
abstract class Task {
    String taskName;

    Task(String taskName) {
        this.taskName = taskName;
    }
    void showTaskName() {
        System.out.println("Task: " + taskName);
    }
    abstract void execute();
}

class PrintTask extends Task {
    PrintTask(String taskName) {
        super(taskName);
    }
    void execute() {
        System.out.println("Executing print task: " + taskName);
    }
}


class DownloadTask extends Task {
    DownloadTask(String taskName) {
        super(taskName);
    }
    void execute() {
        System.out.println("Executing download task: " + taskName);
    }
}


public class Main {
    public static void main(String[] args) {
        Task t1 = new PrintTask("Print Report");
        Task t2 = new DownloadTask("Download File");

        t1.showTaskName();
        t1.execute();

        t2.showTaskName();
        t2.execute();
    }
}
